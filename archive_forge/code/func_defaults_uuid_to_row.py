import collections
import functools
import operator
from ovs.db import data
def defaults_uuid_to_row(atom, base):
    return atom.value