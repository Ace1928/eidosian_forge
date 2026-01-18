import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
import srsly
import typer
from wasabi import msg
from wasabi.util import locale_escape
from ..util import SimpleFrozenDict, SimpleFrozenList, check_spacy_env_vars
from ..util import get_checksum, get_hash, is_cwd, join_command, load_project_config
from ..util import parse_config_overrides, run_command, split_command, working_dir
from .main import COMMAND, PROJECT_FILE, PROJECT_LOCK, Arg, Opt, _get_parent_command
from .main import app
def project_run(project_dir: Path, subcommand: str, *, overrides: Dict[str, Any]=SimpleFrozenDict(), force: bool=False, dry: bool=False, capture: bool=False, skip_requirements_check: bool=False, parent_command: str=COMMAND) -> None:
    """Run a named script defined in the project.yml. If the script is part
    of the default pipeline (defined in the "run" section), DVC is used to
    execute the command, so it can determine whether to rerun it. It then
    calls into "exec" to execute it.

    project_dir (Path): Path to project directory.
    subcommand (str): Name of command to run.
    overrides (Dict[str, Any]): Optional config overrides.
    force (bool): Force re-running, even if nothing changed.
    dry (bool): Perform a dry run and don't execute commands.
    capture (bool): Whether to capture the output and errors of individual commands.
        If False, the stdout and stderr will not be redirected, and if there's an error,
        sys.exit will be called with the return code. You should use capture=False
        when you want to turn over execution to the command, and capture=True
        when you want to run the command more like a function.
    skip_requirements_check (bool): No longer used, deprecated.
    """
    config = load_project_config(project_dir, overrides=overrides)
    commands = {cmd['name']: cmd for cmd in config.get('commands', [])}
    workflows = config.get('workflows', {})
    validate_subcommand(list(commands.keys()), list(workflows.keys()), subcommand)
    if subcommand in workflows:
        msg.info(f"Running workflow '{subcommand}'")
        for cmd in workflows[subcommand]:
            project_run(project_dir, cmd, overrides=overrides, force=force, dry=dry, capture=capture)
    else:
        cmd = commands[subcommand]
        for dep in cmd.get('deps', []):
            if not (project_dir / dep).exists():
                err = f"Missing dependency specified by command '{subcommand}': {dep}"
                err_help = "Maybe you forgot to run the 'project assets' command or a previous step?"
                err_exits = 1 if not dry else None
                msg.fail(err, err_help, exits=err_exits)
        check_spacy_env_vars()
        with working_dir(project_dir) as current_dir:
            msg.divider(subcommand)
            rerun = check_rerun(current_dir, cmd)
            if not rerun and (not force):
                msg.info(f"Skipping '{cmd['name']}': nothing changed")
            else:
                run_commands(cmd['script'], dry=dry, capture=capture)
                if not dry:
                    update_lockfile(current_dir, cmd)