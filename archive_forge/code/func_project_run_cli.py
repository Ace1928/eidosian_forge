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
@app.command('run', context_settings={'allow_extra_args': True, 'ignore_unknown_options': True})
def project_run_cli(ctx: typer.Context, subcommand: str=Arg(None, help=f'Name of command defined in the {PROJECT_FILE}'), project_dir: Path=Arg(Path.cwd(), help='Location of project directory. Defaults to current working directory.', exists=True, file_okay=False), force: bool=Opt(False, '--force', '-F', help='Force re-running steps, even if nothing changed'), dry: bool=Opt(False, '--dry', '-D', help="Perform a dry run and don't execute scripts"), show_help: bool=Opt(False, '--help', help='Show help message and available subcommands')):
    """Run a named command or workflow defined in the project.yml. If a workflow
    name is specified, all commands in the workflow are run, in order. If
    commands define dependencies and/or outputs, they will only be re-run if
    state has changed.

    DOCS: https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run
    """
    parent_command = _get_parent_command(ctx)
    if show_help or not subcommand:
        print_run_help(project_dir, subcommand, parent_command)
    else:
        overrides = parse_config_overrides(ctx.args)
        project_run(project_dir, subcommand, overrides=overrides, force=force, dry=dry, parent_command=parent_command)