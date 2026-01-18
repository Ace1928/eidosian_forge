from __future__ import annotations
import typing
from .columns import Columns
from .constants import WHSettings
from .listbox import ListBox, ListWalker
from .padding import Padding
from .text import Text
from .widget import WidgetWrap
from .wimp import SelectableIcon
def change_child_key(self, oldkey: Hashable, newkey: Hashable) -> None:
    if newkey in self._children:
        raise TreeWidgetError(f'{newkey} is already in use')
    self._children[newkey] = self._children.pop(oldkey)
    self._children[newkey].set_key(newkey)