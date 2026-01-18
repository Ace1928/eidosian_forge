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
def get_dirs_from_args(args: Iterable[str]) -> List[Path]:

    def is_option(x: str) -> bool:
        return x.startswith('-')

    def get_file_part_from_node_id(x: str) -> str:
        return x.split('::')[0]

    def get_dir_from_path(path: Path) -> Path:
        if path.is_dir():
            return path
        return path.parent
    possible_paths = (absolutepath(get_file_part_from_node_id(arg)) for arg in args if not is_option(arg))
    return [get_dir_from_path(path) for path in possible_paths if safe_exists(path)]