from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def endTagCaption(self, token):
    if not self.ignoreEndTagCaption():
        self.tree.generateImpliedEndTags()
        if self.tree.openElements[-1].name != 'caption':
            self.parser.parseError('expected-one-end-tag-but-got-another', {'gotName': 'caption', 'expectedName': self.tree.openElements[-1].name})
        while self.tree.openElements[-1].name != 'caption':
            self.tree.openElements.pop()
        self.tree.openElements.pop()
        self.tree.clearActiveFormattingElements()
        self.parser.phase = self.parser.phases['inTable']
    else:
        assert self.parser.innerHTML
        self.parser.parseError()