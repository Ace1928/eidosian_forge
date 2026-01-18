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
def expand_groups(properties, cdict):
    res = []
    for prop in properties:
        if isinstance(prop, PyGroup):
            cname = prop.ref[1]
            res.extend(cdict[cname].properties[0])
        else:
            res.append(prop)
    return res