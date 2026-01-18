from __future__ import unicode_literals
from collections import OrderedDict
import yaml
from pybtex.database.output import BaseWriter
def process_person_roles(entry):
    for role, persons in entry.persons.items():
        yield (role, list(process_persons(persons)))