import os
import warnings
from typing import Optional, Sequence, Mapping, Callable
from ipywidgets import Dropdown, Text, Select, Button, HTML
from ipywidgets import Layout, GridBox, Box, HBox, VBox, ValueWidget
from .errors import ParentPathError, InvalidFileNameError
from .utils import get_subpaths, get_dir_contents, match_item, strip_parent_path
from .utils import is_valid_filename, get_drive_letters, normalize_path, has_parent_path
@show_only_dirs.setter
def show_only_dirs(self, show_only_dirs: bool) -> None:
    """Set show_only_dirs property value."""
    self._show_only_dirs = show_only_dirs
    self._filename.disabled = self._show_only_dirs
    self._filename.layout.display = (None, 'none')[self._show_only_dirs]
    self._gb.layout.children = [self._pathlist, self._dircontent]
    if not self._show_only_dirs:
        self._gb.layout.children.insert(1, self._filename)
    self._gb.layout.grid_template_areas = "\n            'pathlist {}'\n            'dircontent dircontent'\n            ".format(('filename', 'pathlist')[self._show_only_dirs])
    self.reset()