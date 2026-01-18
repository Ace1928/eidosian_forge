import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Union
from collections import OrderedDict
import yaml
def parse_and_validate_pip(pip: Union[str, List[str], Dict]) -> Optional[Dict]:
    """Parses and validates a user-provided 'pip' option.

    The value of the input 'pip' field can be one of two cases:
        1) A List[str] describing the requirements. This is passed through.
        2) A string pointing to a local requirements file. In this case, the
           file contents will be read split into a list.
        3) A python dictionary that has three fields:
            a) packages (required, List[str]): a list of pip packages, it same as 1).
            b) pip_check (optional, bool): whether to enable pip check at the end of pip
               install, default to False.
            c) pip_version (optional, str): the version of pip, ray will spell
               the package name 'pip' in front of the `pip_version` to form the final
               requirement string, the syntax of a requirement specifier is defined in
               full in PEP 508.

    The returned parsed value will be a list of pip packages. If a Ray library
    (e.g. "ray[serve]") is specified, it will be deleted and replaced by its
    dependencies (e.g. "uvicorn", "requests").
    """
    assert pip is not None

    def _handle_local_pip_requirement_file(pip_file: str):
        pip_path = Path(pip_file)
        if not pip_path.is_file():
            raise ValueError(f'{pip_path} is not a valid file')
        return pip_path.read_text().strip().split('\n')
    result = None
    if sys.platform == 'win32':
        logger.warning('runtime environment support is experimental on Windows. If you run into issues please file a report at https://github.com/ray-project/ray/issues.')
    if isinstance(pip, str):
        pip_list = _handle_local_pip_requirement_file(pip)
        result = dict(packages=pip_list, pip_check=False)
    elif isinstance(pip, list) and all((isinstance(dep, str) for dep in pip)):
        result = dict(packages=pip, pip_check=False)
    elif isinstance(pip, dict):
        if set(pip.keys()) - {'packages', 'pip_check', 'pip_version'}:
            raise ValueError(f"runtime_env['pip'] can only have these fields: packages, pip_check and pip_version, but got: {list(pip.keys())}")
        if 'pip_check' in pip and (not isinstance(pip['pip_check'], bool)):
            raise TypeError(f"runtime_env['pip']['pip_check'] must be of type bool, got {type(pip['pip_check'])}")
        if 'pip_version' in pip:
            if not isinstance(pip['pip_version'], str):
                raise TypeError(f"runtime_env['pip']['pip_version'] must be of type str, got {type(pip['pip_version'])}")
        result = pip.copy()
        result['pip_check'] = pip.get('pip_check', False)
        if 'packages' not in pip:
            raise ValueError(f"runtime_env['pip'] must include field 'packages', but got {pip}")
        elif isinstance(pip['packages'], str):
            result['packages'] = _handle_local_pip_requirement_file(pip['packages'])
        elif not isinstance(pip['packages'], list):
            raise ValueError(f"runtime_env['pip']['packages'] must be of type str of list, got: {type(pip['packages'])}")
    else:
        raise TypeError(f"runtime_env['pip'] must be of type str or List[str], got {type(pip)}")
    result['packages'] = list(OrderedDict.fromkeys(result['packages']))
    if len(result['packages']) == 0:
        result = None
    logger.debug(f'Rewrote runtime_env `pip` field from {pip} to {result}.')
    return result