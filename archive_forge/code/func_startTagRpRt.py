from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def startTagRpRt(self, token):
    if self.tree.elementInScope('ruby'):
        self.tree.generateImpliedEndTags()
        if self.tree.openElements[-1].name != 'ruby':
            self.parser.parseError()
    self.tree.insertElement(token)