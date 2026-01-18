from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def startTagListItem(self, token):
    self.parser.framesetOK = False
    stopNamesMap = {'li': ['li'], 'dt': ['dt', 'dd'], 'dd': ['dt', 'dd']}
    stopNames = stopNamesMap[token['name']]
    for node in reversed(self.tree.openElements):
        if node.name in stopNames:
            self.parser.phase.processEndTag(impliedTagToken(node.name, 'EndTag'))
            break
        if node.nameTuple in specialElements and node.name not in ('address', 'div', 'p'):
            break
    if self.tree.elementInScope('p', variant='button'):
        self.parser.phase.processEndTag(impliedTagToken('p', 'EndTag'))
    self.tree.insertElement(token)