from __future__ import unicode_literals
from pybtex.style.sorting import BaseSortingStyle
def persons_key(self, persons):
    return '   '.join((self.person_key(person) for person in persons))