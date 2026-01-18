import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def makeXMLTags(tagStr):
    """Helper to construct opening and closing tag expressions for XML, given a tag name"""
    return _makeTags(tagStr, True)