import importlib
import logging
import re
from io import StringIO
from xml.dom import minidom
from xml.parsers.expat import ExpatError
from rdkit.Chem import Mol
def is_molecule_image(s):
    result = False
    try:
        xml = minidom.parseString(s)
        root_node = xml.firstChild
        if root_node.nodeName in ['svg', 'img', 'div'] and 'data-content' in root_node.attributes.keys() and (root_node.attributes['data-content'].value == 'rdkit/molecule'):
            result = True
    except ExpatError:
        pass
    return result