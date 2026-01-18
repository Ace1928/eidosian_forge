import os
import re
from pathlib import Path
from typing import Union
from urllib.parse import unquote, urlparse
def is_relative(root: Union[str, Path], path: Union[str, Path]) -> bool:
    """Return if path is relative to root"""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False