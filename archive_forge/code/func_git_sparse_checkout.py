import shutil
from pathlib import Path
from typing import Tuple
from wasabi import msg
from .commands import run_command
from .filesystem import is_subpath_of, make_tempdir
def git_sparse_checkout(repo, subpath, dest, branch):
    with make_tempdir() as tmp_dir:
        cmd = f'git clone {repo} {tmp_dir} --no-checkout --depth 1 -b {branch} --filter=blob:none'
        run_command(cmd)
        cmd = f'git -C {tmp_dir} rev-list --objects --all --missing=print -- {subpath}'
        ret = run_command(cmd, capture=True)
        git_repo = _http_to_git(repo)
        missings = ' '.join([x[1:] for x in ret.stdout.split() if x.startswith('?')])
        if not missings:
            err = f"Could not find any relevant files for '{subpath}'. Did you specify a correct and complete path within repo '{repo}' and branch {branch}?"
            msg.fail(err, exits=1)
        cmd = f'git -C {tmp_dir} fetch-pack {git_repo} {missings}'
        run_command(cmd, capture=True)
        cmd = f'git -C {tmp_dir} checkout {branch} {subpath}'
        run_command(cmd, capture=True)
        source_path = tmp_dir / Path(subpath)
        if not is_subpath_of(tmp_dir, source_path):
            err = f"'{subpath}' is a path outside of the cloned repository."
            msg.fail(err, repo, exits=1)
        shutil.move(str(source_path), str(dest))