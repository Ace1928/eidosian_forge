import re
from uritemplate.orderedset import OrderedSet
from uritemplate.variable import URIVariable
def replace_all(match):
    return expanded.get(match.groups()[0], '')