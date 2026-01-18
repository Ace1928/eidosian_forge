from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def startTagTableOther(self, token):
    if self.tree.elementInScope('td', variant='table') or self.tree.elementInScope('th', variant='table'):
        self.closeCell()
        return token
    else:
        assert self.parser.innerHTML
        self.parser.parseError()