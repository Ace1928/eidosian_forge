from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def processSpaceCharactersDropNewline(self, token):
    data = token['data']
    self.processSpaceCharacters = self.processSpaceCharactersNonPre
    if data.startswith('\n') and self.tree.openElements[-1].name in ('pre', 'listing', 'textarea') and (not self.tree.openElements[-1].hasContent()):
        data = data[1:]
    if data:
        self.tree.reconstructActiveFormattingElements()
        self.tree.insertText(data)