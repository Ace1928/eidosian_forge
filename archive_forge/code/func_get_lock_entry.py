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
def get_lock_entry(project_dir: Path, command: Dict[str, Any], *, parent_command: str=COMMAND) -> Dict[str, Any]:
    """Get a lockfile entry for a given command. An entry includes the command,
    the script (command steps) and a list of dependencies and outputs with
    their paths and file hashes, if available. The format is based on the
    dvc.lock files, to keep things consistent.

    project_dir (Path): The current project directory.
    command (Dict[str, Any]): The command, as defined in the project.yml.
    RETURNS (Dict[str, Any]): The lockfile entry.
    """
    deps = get_fileinfo(project_dir, command.get('deps', []))
    outs = get_fileinfo(project_dir, command.get('outputs', []))
    outs_nc = get_fileinfo(project_dir, command.get('outputs_no_cache', []))
    return {'cmd': f'{parent_command} run {command['name']}', 'script': command['script'], 'deps': deps, 'outs': [*outs, *outs_nc]}