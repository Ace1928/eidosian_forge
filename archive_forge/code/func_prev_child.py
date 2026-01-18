from __future__ import annotations
import typing
from .columns import Columns
from .constants import WHSettings
from .listbox import ListBox, ListWalker
from .padding import Padding
from .text import Text
from .widget import WidgetWrap
from .wimp import SelectableIcon
def prev_child(self, key: Hashable) -> TreeNode | None:
    """Return the previous child node in index order from the given key."""
    index = self.get_child_index(key)
    if index is None:
        return None
    child_keys = self.get_child_keys()
    index -= 1
    if index >= 0:
        return self.get_child_node(child_keys[index])
    return None