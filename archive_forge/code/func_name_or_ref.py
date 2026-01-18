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
def name_or_ref(elem, top):
    try:
        namespace, name = _namespace_and_tag(elem, elem.ref, top)
        if namespace and elem.xmlns_map[namespace] == top.target_namespace:
            return name
        else:
            return elem.ref
    except AttributeError:
        return elem.name