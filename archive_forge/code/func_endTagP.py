from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def endTagP(self, token):
    if not self.tree.elementInScope('p', variant='button'):
        self.startTagCloseP(impliedTagToken('p', 'StartTag'))
        self.parser.parseError('unexpected-end-tag', {'name': 'p'})
        self.endTagP(impliedTagToken('p', 'EndTag'))
    else:
        self.tree.generateImpliedEndTags('p')
        if self.tree.openElements[-1].name != 'p':
            self.parser.parseError('unexpected-end-tag', {'name': 'p'})
        node = self.tree.openElements.pop()
        while node.name != 'p':
            node = self.tree.openElements.pop()