import time
from itertools import chain
from xml.dom import minidom
import xml.etree.ElementTree as ET
def get_KGML(self):
    """Return the pathway as a string in prettified KGML format."""
    header = '\n'.join(['<?xml version="1.0"?>', '<!DOCTYPE pathway SYSTEM "http://www.genome.jp/kegg/xml/KGML_v0.7.2_.dtd">', f'<!-- Created by KGML_Pathway.py {time.asctime()} -->'])
    rough_xml = header + ET.tostring(self.element, 'utf-8').decode()
    reparsed = minidom.parseString(rough_xml)
    return reparsed.toprettyxml(indent='  ')