import collections
import functools
import hashlib
import itertools
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from urllib import parse as urlparse
import yaql
from yaql.language import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import function
def get_path_component(collection, key):
    if not isinstance(collection, (collections.abc.Mapping, collections.abc.Sequence)):
        raise TypeError(_('"%s" can\'t traverse path') % self.fn_name)
    if not isinstance(key, (str, int)):
        raise TypeError(_('Path components in "%s" must be strings') % self.fn_name)
    if isinstance(collection, collections.abc.Sequence) and isinstance(key, str):
        try:
            key = int(key)
        except ValueError:
            raise TypeError(_("Path components in '%s' must be a string that can be parsed into an integer.") % self.fn_name)
    return collection[key]