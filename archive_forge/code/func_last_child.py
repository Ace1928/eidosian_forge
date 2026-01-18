from __future__ import annotations
import typing
from .columns import Columns
from .constants import WHSettings
from .listbox import ListBox, ListWalker
from .padding import Padding
from .text import Text
from .widget import WidgetWrap
from .wimp import SelectableIcon
def last_child(self) -> TreeWidget | None:
    """Return last child if expanded."""
    if self.is_leaf or not self.expanded:
        return None
    if self._node.has_children():
        last_child = self._node.get_last_child().get_widget()
    else:
        return None
    last_descendant = last_child.last_child()
    if last_descendant is None:
        return last_child
    return last_descendant