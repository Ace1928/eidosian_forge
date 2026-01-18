import json
import xml.etree.ElementTree as ET
from langcodes.util import data_filename
from langcodes.registry_parser import parse_registry
def read_iana_registry_suppress_scripts():
    scripts = {}
    for entry in parse_registry():
        if entry['Type'] == 'language' and 'Suppress-Script' in entry:
            scripts[entry['Subtag']] = entry['Suppress-Script']
    return scripts