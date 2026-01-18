import re
from distutils.version import LooseVersion
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Union
from pkg_resources import Requirement, yield_lines
def load_requirements(path_dir: str, file_name: str='base.txt', unfreeze: str='all') -> List[str]:
    """Load requirements from a file.

    >>> import os
    >>> from lightning_utilities import _PROJECT_ROOT
    >>> path_req = os.path.join(_PROJECT_ROOT, "requirements")
    >>> load_requirements(path_req, "docs.txt", unfreeze="major")  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['sphinx<6.0,>=4.0', ...]

    """
    if unfreeze not in {'none', 'major', 'all'}:
        raise ValueError(f'unsupported option of "{unfreeze}"')
    path = Path(path_dir) / file_name
    if not path.exists():
        raise FileNotFoundError(f'missing file for {(path_dir, file_name, path)}')
    text = path.read_text()
    return [req.adjust(unfreeze) for req in _parse_requirements(text)]