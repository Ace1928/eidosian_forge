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
def resource_description(name, info):
    if not with_description:
        return name
    rsrc_cls = info.get_class()
    if rsrc_cls is None:
        rsrc_cls = heat.engine.resource.Resource
    return {'resource_type': name, 'description': rsrc_cls.getdoc()}