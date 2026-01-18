import collections
import copy
import functools
import itertools
import operator
from heat.common import exception
from heat.engine import function
from heat.engine import properties
def properties_changed(self):
    """Return True if the resource properties have changed."""
    return self.old_defn._properties != self.new_defn._properties