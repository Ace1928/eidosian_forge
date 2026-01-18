from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def endTagSelect(self, token):
    if self.tree.elementInScope('select', variant='select'):
        node = self.tree.openElements.pop()
        while node.name != 'select':
            node = self.tree.openElements.pop()
        self.parser.resetInsertionMode()
    else:
        assert self.parser.innerHTML
        self.parser.parseError()