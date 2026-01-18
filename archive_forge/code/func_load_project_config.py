import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import srsly
from click import NoSuchOption
from click.parser import split_arg_string
from confection import Config
from wasabi import msg
from ..cli.main import PROJECT_FILE
from ..schemas import ProjectConfigSchema, validate
from .environment import ENV_VARS
from .frozen import SimpleFrozenDict
from .logging import logger
from .validation import show_validation_error, validate_project_commands
def load_project_config(path: Path, interpolate: bool=True, overrides: Dict[str, Any]=SimpleFrozenDict()) -> Dict[str, Any]:
    """Load the project.yml file from a directory and validate it. Also make
    sure that all directories defined in the config exist.

    path (Path): The path to the project directory.
    interpolate (bool): Whether to substitute project variables.
    overrides (Dict[str, Any]): Optional config overrides.
    RETURNS (Dict[str, Any]): The loaded project.yml.
    """
    config_path = path / PROJECT_FILE
    if not config_path.exists():
        msg.fail(f"Can't find {PROJECT_FILE}", config_path, exits=1)
    invalid_err = f'Invalid {PROJECT_FILE}. Double-check that the YAML is correct.'
    try:
        config = srsly.read_yaml(config_path)
    except ValueError as e:
        msg.fail(invalid_err, e, exits=1)
    errors = validate(ProjectConfigSchema, config)
    if errors:
        msg.fail(invalid_err)
        print('\n'.join(errors))
        sys.exit(1)
    validate_project_commands(config)
    if interpolate:
        err = f'{PROJECT_FILE} validation error'
        with show_validation_error(title=err, hint_fill=False):
            config = substitute_project_variables(config, overrides)
    for subdir in config.get('directories', []):
        dir_path = path / subdir
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
    return config