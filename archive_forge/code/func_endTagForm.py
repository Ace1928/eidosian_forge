from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def endTagForm(self, token):
    node = self.tree.formPointer
    self.tree.formPointer = None
    if node is None or not self.tree.elementInScope(node):
        self.parser.parseError('unexpected-end-tag', {'name': 'form'})
    else:
        self.tree.generateImpliedEndTags()
        if self.tree.openElements[-1] != node:
            self.parser.parseError('end-tag-too-early-ignored', {'name': 'form'})
        self.tree.openElements.remove(node)