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
def rm_duplicates(properties):
    keys = []
    clist = []
    for prop in properties:
        if prop.name in keys:
            continue
        else:
            clist.append(prop)
            keys.append(prop.name)
    return clist