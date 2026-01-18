import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Union
from collections import OrderedDict
import yaml
def parse_and_validate_conda(conda: Union[str, dict]) -> Union[str, dict]:
    """Parses and validates a user-provided 'conda' option.

    Conda can be one of three cases:
        1) A dictionary describing the env. This is passed through directly.
        2) A string referring to the name of a preinstalled conda env.
        3) A string pointing to a local conda YAML file. This is detected
           by looking for a '.yaml' or '.yml' suffix. In this case, the file
           will be read as YAML and passed through as a dictionary.
    """
    assert conda is not None
    if sys.platform == 'win32':
        logger.warning('runtime environment support is experimental on Windows. If you run into issues please file a report at https://github.com/ray-project/ray/issues.')
    result = None
    if isinstance(conda, str):
        yaml_file = Path(conda)
        if yaml_file.suffix in ('.yaml', '.yml'):
            if not yaml_file.is_file():
                raise ValueError(f"Can't find conda YAML file {yaml_file}.")
            try:
                result = yaml.safe_load(yaml_file.read_text())
            except Exception as e:
                raise ValueError(f'Failed to read conda file {yaml_file}: {e}.')
        else:
            result = conda
    elif isinstance(conda, dict):
        result = conda
    else:
        raise TypeError(f"runtime_env['conda'] must be of type str or dict, got {type(conda)}.")
    return result