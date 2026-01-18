from __future__ import unicode_literals
from xml.etree import ElementTree as ET
from pybtex.database import Entry, Person
from pybtex.database.input import BaseParser
def process_person(person_entry, role):
    persons = person_entry.findall(bibtexns + 'person')
    if persons:
        for person in persons:
            process_person(person, role)
    else:
        text = person_entry.text.strip()
        if text:
            e.add_person(Person(text), role)
        else:
            names = {}
            for name in person_entry:
                names[remove_ns(name.tag)] = name.text
            e.add_person(Person(**names), role)