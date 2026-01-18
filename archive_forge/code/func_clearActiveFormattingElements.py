from __future__ import absolute_import, division, unicode_literals
from six import text_type
from ..constants import scopingElements, tableInsertModeElements, namespaces
def clearActiveFormattingElements(self):
    entry = self.activeFormattingElements.pop()
    while self.activeFormattingElements and entry != Marker:
        entry = self.activeFormattingElements.pop()