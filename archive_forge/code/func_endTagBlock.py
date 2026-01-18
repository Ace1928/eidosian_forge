from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def endTagBlock(self, token):
    if token['name'] == 'pre':
        self.processSpaceCharacters = self.processSpaceCharactersNonPre
    inScope = self.tree.elementInScope(token['name'])
    if inScope:
        self.tree.generateImpliedEndTags()
    if self.tree.openElements[-1].name != token['name']:
        self.parser.parseError('end-tag-too-early', {'name': token['name']})
    if inScope:
        node = self.tree.openElements.pop()
        while node.name != token['name']:
            node = self.tree.openElements.pop()