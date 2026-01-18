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
def remove_resources_except(self, resource_name):
    ress = self._registry['resources']
    new_resources = {}
    for name, res in ress.items():
        if fnmatch.fnmatchcase(resource_name, name):
            new_resources.update(res)
    if resource_name in ress:
        new_resources.update(ress[resource_name])
    self._registry['resources'] = new_resources