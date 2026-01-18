import abc
import collections
import itertools
import weakref
from heat.common import exception
from heat.common.i18n import _
def res_dep_attrs(resource_name):
    return zip(itertools.repeat(resource_name), self.dep_attrs(resource_name))