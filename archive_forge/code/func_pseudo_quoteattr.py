import sys
import os
import re
import warnings
import types
import unicodedata
def pseudo_quoteattr(value):
    """Quote attributes for pseudo-xml"""
    return '"%s"' % value