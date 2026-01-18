import os
import subprocess
import contextlib
import functools
import tempfile
import shutil
import operator
import warnings
@contextlib.contextmanager
def tarball_context(url, target_dir=None, runner=None, pushd=pushd):
    """
    Get a tarball, extract it, change to that directory, yield, then
    clean up.
    `runner` is the function to invoke commands.
    `pushd` is a context manager for changing the directory.
    """
    if target_dir is None:
        target_dir = os.path.basename(url).replace('.tar.gz', '').replace('.tgz', '')
    if runner is None:
        runner = functools.partial(subprocess.check_call, shell=True)
    else:
        warnings.warn('runner parameter is deprecated', DeprecationWarning)
    runner('mkdir {target_dir}'.format(**vars()))
    try:
        getter = 'wget {url} -O -'
        extract = 'tar x{compression} --strip-components=1 -C {target_dir}'
        cmd = ' | '.join((getter, extract))
        runner(cmd.format(compression=infer_compression(url), **vars()))
        with pushd(target_dir):
            yield target_dir
    finally:
        runner('rm -Rf {target_dir}'.format(**vars()))