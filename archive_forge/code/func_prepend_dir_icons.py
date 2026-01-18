import fnmatch
import os
import string
import sys
from typing import List, Sequence, Iterable, Optional
from .errors import InvalidPathError
def prepend_dir_icons(dir_list: Iterable[str], dir_icon: str, dir_icon_append: bool=False) -> List[str]:
    """Prepend unicode folder icon to directory names."""
    if dir_icon_append:
        str_ = [dirname + f'{dir_icon}' for dirname in dir_list]
    else:
        str_ = [f'{dir_icon}' + dirname for dirname in dir_list]
    return str_