from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def startTagInput(self, token):
    self.parser.parseError('unexpected-input-in-select')
    if self.tree.elementInScope('select', variant='select'):
        self.endTagSelect(impliedTagToken('select'))
        return token
    else:
        assert self.parser.innerHTML