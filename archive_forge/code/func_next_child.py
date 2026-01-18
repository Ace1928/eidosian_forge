from __future__ import annotations
import typing
from .columns import Columns
from .constants import WHSettings
from .listbox import ListBox, ListWalker
from .padding import Padding
from .text import Text
from .widget import WidgetWrap
from .wimp import SelectableIcon
def next_child(self, key: Hashable) -> TreeNode | None:
    """Return the next child node in index order from the given key."""
    index = self.get_child_index(key)
    if index is None:
        return None
    index += 1
    child_keys = self.get_child_keys()
    if index < len(child_keys):
        return self.get_child_node(child_keys[index])
    return None