from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def closeCell(self):
    if self.tree.elementInScope('td', variant='table'):
        self.endTagTableCell(impliedTagToken('td'))
    elif self.tree.elementInScope('th', variant='table'):
        self.endTagTableCell(impliedTagToken('th'))