from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def resetInsertionMode(self):
    last = False
    newModes = {'select': 'inSelect', 'td': 'inCell', 'th': 'inCell', 'tr': 'inRow', 'tbody': 'inTableBody', 'thead': 'inTableBody', 'tfoot': 'inTableBody', 'caption': 'inCaption', 'colgroup': 'inColumnGroup', 'table': 'inTable', 'head': 'inBody', 'body': 'inBody', 'frameset': 'inFrameset', 'html': 'beforeHead'}
    for node in self.tree.openElements[::-1]:
        nodeName = node.name
        new_phase = None
        if node == self.tree.openElements[0]:
            assert self.innerHTML
            last = True
            nodeName = self.innerHTML
        if nodeName in ('select', 'colgroup', 'head', 'html'):
            assert self.innerHTML
        if not last and node.namespace != self.tree.defaultNamespace:
            continue
        if nodeName in newModes:
            new_phase = self.phases[newModes[nodeName]]
            break
        elif last:
            new_phase = self.phases['inBody']
            break
    self.phase = new_phase