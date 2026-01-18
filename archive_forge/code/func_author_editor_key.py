from __future__ import unicode_literals
from pybtex.style.sorting import BaseSortingStyle
def author_editor_key(self, entry):
    if entry.persons.get('author'):
        return self.persons_key(entry.persons['author'])
    elif entry.persons.get('editor'):
        return self.persons_key(entry.persons['editor'])
    else:
        return ''