import copy
import datetime
import keyword
import re
import unicodedata
import warnings
def python_name(name):
    """ Attempt to make a valid Python identifier out of a name.
    """
    if len(name) > 0:
        name = name.replace(' ', '_').lower()
        if keyword.iskeyword(name):
            name = '_' + name
        if name[0].isdigit():
            name = '_' + name
    return name