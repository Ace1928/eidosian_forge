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
def child_spec(self, target_namespace, prop, mod, typ, lista):
    if mod:
        namesp = external_namespace(self.root.modul[mod])
        pkey = f'{{{namesp}}}{prop.name}'
        typ = f'{mod}.{typ}'
    else:
        pkey = f'{{{target_namespace}}}{prop.name}'
    if lista:
        return f"c_children['{pkey}'] = ('{prop.pyname}', [{typ}])"
    else:
        return f"c_children['{pkey}'] = ('{prop.pyname}', {typ})"