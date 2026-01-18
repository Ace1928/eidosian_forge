import collections.abc
import copy
import functools
from dogpile.cache import region as dp_region
from oslo_utils import uuidutils
from oslo_cache.backends import mongo
from oslo_cache import core
from oslo_cache import exception
from oslo_cache.tests import test_cache
def find_and_modify(self, spec, document, upsert=False, **kwargs):
    self.update(spec, document, upsert, **kwargs)