import collections
import fnmatch
import glob
import itertools
import os.path
import re
import weakref
from oslo_config import cfg
from oslo_log import log
from heat.common import environment_format as env_fmt
from heat.common import exception
from heat.common.i18n import _
from heat.common import policy
from heat.engine import support
def iterable_by(self, resource_type, resource_name=None):
    is_templ_type = resource_type.endswith(('.yaml', '.template'))
    if self.global_registry is not None and is_templ_type:
        if resource_type not in self._registry:
            res = ResourceInfo(self, [resource_type], None)
            self._register_info([resource_type], res)
        yield self._registry[resource_type]
    if resource_name:
        impl = self._registry['resources'].get(resource_name)
        if impl and resource_type in impl:
            yield impl[resource_type]
    impl = self._registry.get(resource_type)
    if impl:
        yield impl

    def is_a_glob(resource_type):
        return resource_type.endswith('*')
    globs = filter(is_a_glob, iter(self._registry))
    for pattern in globs:
        if self._registry[pattern].matches(resource_type):
            yield self._registry[pattern]