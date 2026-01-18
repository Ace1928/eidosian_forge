import json
import xml.etree.ElementTree as ET
from langcodes.util import data_filename
from langcodes.registry_parser import parse_registry
def read_iana_registry_macrolanguages():
    macros = {}
    for entry in parse_registry():
        if entry['Type'] == 'language' and 'Macrolanguage' in entry:
            macros[entry['Subtag']] = entry['Macrolanguage']
    return macros