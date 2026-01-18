from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def endTagTableRowGroup(self, token):
    if self.tree.elementInScope(token['name'], variant='table'):
        self.endTagTr(impliedTagToken('tr'))
        return token
    else:
        self.parser.parseError()