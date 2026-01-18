import json
import xml.etree.ElementTree as ET
from langcodes.util import data_filename
from langcodes.registry_parser import parse_registry
def read_iana_registry_scripts():
    scripts = set()
    for entry in parse_registry():
        if entry['Type'] == 'script':
            scripts.add(entry['Subtag'])
    return scripts