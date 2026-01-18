from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def endTagTableCell(self, token):
    if self.tree.elementInScope(token['name'], variant='table'):
        self.tree.generateImpliedEndTags(token['name'])
        if self.tree.openElements[-1].name != token['name']:
            self.parser.parseError('unexpected-cell-end-tag', {'name': token['name']})
            while True:
                node = self.tree.openElements.pop()
                if node.name == token['name']:
                    break
        else:
            self.tree.openElements.pop()
        self.tree.clearActiveFormattingElements()
        self.parser.phase = self.parser.phases['inRow']
    else:
        self.parser.parseError('unexpected-end-tag', {'name': token['name']})