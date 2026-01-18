import os
from pathlib import Path
import sys
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
import iniconfig
from .exceptions import UsageError
from _pytest.outcomes import fail
from _pytest.pathlib import absolutepath
from _pytest.pathlib import commonpath
from _pytest.pathlib import safe_exists
def locate_config(invocation_dir: Path, args: Iterable[Path]) -> Tuple[Optional[Path], Optional[Path], Dict[str, Union[str, List[str]]]]:
    """Search in the list of arguments for a valid ini-file for pytest,
    and return a tuple of (rootdir, inifile, cfg-dict)."""
    config_names = ['pytest.ini', '.pytest.ini', 'pyproject.toml', 'tox.ini', 'setup.cfg']
    args = [x for x in args if not str(x).startswith('-')]
    if not args:
        args = [invocation_dir]
    found_pyproject_toml: Optional[Path] = None
    for arg in args:
        argpath = absolutepath(arg)
        for base in (argpath, *argpath.parents):
            for config_name in config_names:
                p = base / config_name
                if p.is_file():
                    if p.name == 'pyproject.toml' and found_pyproject_toml is None:
                        found_pyproject_toml = p
                    ini_config = load_config_dict_from_file(p)
                    if ini_config is not None:
                        return (base, p, ini_config)
    if found_pyproject_toml is not None:
        return (found_pyproject_toml.parent, found_pyproject_toml, {})
    return (None, None, {})