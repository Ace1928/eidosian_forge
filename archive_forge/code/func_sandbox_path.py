import os
import warnings
from typing import Optional, Sequence, Mapping, Callable
from ipywidgets import Dropdown, Text, Select, Button, HTML
from ipywidgets import Layout, GridBox, Box, HBox, VBox, ValueWidget
from .errors import ParentPathError, InvalidFileNameError
from .utils import get_subpaths, get_dir_contents, match_item, strip_parent_path
from .utils import is_valid_filename, get_drive_letters, normalize_path, has_parent_path
@sandbox_path.setter
def sandbox_path(self, sandbox_path: str) -> None:
    """Set the sandbox_path."""
    if sandbox_path and (not has_parent_path(self._default_path, normalize_path(sandbox_path))):
        raise ParentPathError(self._default_path, sandbox_path)
    self._sandbox_path = normalize_path(sandbox_path) if sandbox_path is not None else None
    self.reset()