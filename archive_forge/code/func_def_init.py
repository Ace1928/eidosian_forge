import errno
import getopt
import importlib
import re
import sys
import time
import types
from xml.etree import ElementTree as ElementTree
import saml2
from saml2 import SamlBase
def def_init(imports, attributes):
    indent = INDENT + INDENT
    indent3 = INDENT + INDENT + INDENT
    line = [f'{INDENT}def __init__(self,']
    for elem in attributes:
        if elem[0] in PROTECTED_KEYWORDS:
            _name = elem[0] + '_'
        else:
            _name = elem[0]
        if elem[2]:
            line.append(f"{indent3}{_name}='{elem[2]}',")
        else:
            line.append(f'{indent3}{_name}={elem[2]},')
    for _, elems in imports.items():
        for elem in elems:
            if elem in PROTECTED_KEYWORDS:
                _name = elem + '_'
            else:
                _name = elem
            line.append(f'{indent3}{_name}=None,')
    line.append(f'{indent3}text=None,')
    line.append(f'{indent3}extension_elements=None,')
    line.append(f'{indent3}extension_attributes=None,')
    line.append(f'{indent}):')
    return line