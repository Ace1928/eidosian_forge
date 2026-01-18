from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def startTagHeading(self, token):
    if self.tree.elementInScope('p', variant='button'):
        self.endTagP(impliedTagToken('p'))
    if self.tree.openElements[-1].name in headingElements:
        self.parser.parseError('unexpected-start-tag', {'name': token['name']})
        self.tree.openElements.pop()
    self.tree.insertElement(token)