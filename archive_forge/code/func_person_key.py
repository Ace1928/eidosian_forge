from __future__ import unicode_literals
from pybtex.style.sorting import BaseSortingStyle
def person_key(self, person):
    return '  '.join((' '.join(person.prelast_names + person.last_names), ' '.join(person.first_names + person.middle_names), ' '.join(person.lineage_names))).lower()