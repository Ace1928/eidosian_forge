import collections
import copy
import functools
import itertools
import operator
from heat.common import exception
from heat.engine import function
from heat.engine import properties
def metadata_changed(self):
    """Return True if the resource metadata has changed."""
    return self.old_defn._metadata != self.new_defn._metadata