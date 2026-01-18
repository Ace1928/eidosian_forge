import os
import warnings
from typing import Optional, Sequence, Mapping, Callable
from ipywidgets import Dropdown, Text, Select, Button, HTML
from ipywidgets import Layout, GridBox, Box, HBox, VBox, ValueWidget
from .errors import ParentPathError, InvalidFileNameError
from .utils import get_subpaths, get_dir_contents, match_item, strip_parent_path
from .utils import is_valid_filename, get_drive_letters, normalize_path, has_parent_path
@dir_icon_append.setter
def dir_icon_append(self, dir_icon_append: bool) -> None:
    """Prepend or append the dir icon."""
    self._dir_icon_append = dir_icon_append
    self.refresh()