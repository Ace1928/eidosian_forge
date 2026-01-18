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
def do_child(self, elem):
    for child in elem:
        self.parts.append(evaluate(child.tag, child))