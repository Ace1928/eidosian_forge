from __future__ import absolute_import, division, unicode_literals
from six import text_type
from ..constants import scopingElements, tableInsertModeElements, namespaces
def reconstructActiveFormattingElements(self):
    if not self.activeFormattingElements:
        return
    i = len(self.activeFormattingElements) - 1
    entry = self.activeFormattingElements[i]
    if entry == Marker or entry in self.openElements:
        return
    while entry != Marker and entry not in self.openElements:
        if i == 0:
            i = -1
            break
        i -= 1
        entry = self.activeFormattingElements[i]
    while True:
        i += 1
        entry = self.activeFormattingElements[i]
        clone = entry.cloneNode()
        element = self.insertElement({'type': 'StartTag', 'name': clone.name, 'namespace': clone.namespace, 'data': clone.attributes})
        self.activeFormattingElements[i] = element
        if element == self.activeFormattingElements[-1]:
            break