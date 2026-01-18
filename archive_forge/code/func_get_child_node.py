from __future__ import annotations
import typing
from .columns import Columns
from .constants import WHSettings
from .listbox import ListBox, ListWalker
from .padding import Padding
from .text import Text
from .widget import WidgetWrap
from .wimp import SelectableIcon
def get_child_node(self, key, reload: bool=False) -> TreeNode:
    """Return the child node for a given key. Create if necessary."""
    if key not in self._children or reload:
        self._children[key] = self.load_child_node(key)
    return self._children[key]