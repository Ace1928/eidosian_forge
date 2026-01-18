import collections
import itertools
import sys
from oslo_config import cfg
from oslo_log import log
from heat.common import plugin_loader
def map_to_modules(self, function):
    """Iterate over the results of calling a function on every module."""
    return map(function, self.modules)