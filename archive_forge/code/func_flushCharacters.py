from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def flushCharacters(self):
    data = ''.join([item['data'] for item in self.characterTokens])
    if any([item not in spaceCharacters for item in data]):
        token = {'type': tokenTypes['Characters'], 'data': data}
        self.parser.phases['inTable'].insertText(token)
    elif data:
        self.tree.insertText(data)
    self.characterTokens = []