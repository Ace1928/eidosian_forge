import re
from uritemplate.orderedset import OrderedSet
from uritemplate.variable import URIVariable
def replace_partial(match):
    match = match.groups()[0]
    var = '{%s}' % match
    return expanded.get(match) or var