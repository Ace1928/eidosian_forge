from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def endTagTr(self, token):
    if not self.ignoreEndTagTr():
        self.clearStackToTableRowContext()
        self.tree.openElements.pop()
        self.parser.phase = self.parser.phases['inTableBody']
    else:
        assert self.parser.innerHTML
        self.parser.parseError()