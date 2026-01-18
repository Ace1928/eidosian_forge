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
def leading_uppercase(string):
    try:
        return string[0].upper() + string[1:]
    except IndexError:
        return string
    except TypeError:
        return ''