from __future__ import annotations
import typing
from .columns import Columns
from .constants import WHSettings
from .listbox import ListBox, ListWalker
from .padding import Padding
from .text import Text
from .widget import WidgetWrap
from .wimp import SelectableIcon
def get_first_child(self) -> TreeNode:
    """Return the first TreeNode in the directory."""
    child_keys = self.get_child_keys()
    return self.get_child_node(child_keys[0])