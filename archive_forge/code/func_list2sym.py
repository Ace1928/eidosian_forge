import html
import re
from collections import defaultdict
def list2sym(lst):
    """
    Convert a list of strings into a canonical symbol.
    :type lst: list
    :return: a Unicode string without whitespace
    :rtype: unicode
    """
    sym = _join(lst, '_', untag=True)
    sym = sym.lower()
    ENT = re.compile('&(\\w+?);')
    sym = ENT.sub(descape_entity, sym)
    sym = sym.replace('.', '')
    return sym