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
def dep_attrs(self, resource_name):
    if self._res_name() == resource_name:
        try:
            attrs = [self._attr_path()]
        except Exception as exc:
            LOG.debug('Ignoring exception calculating required attributes: %s %s', type(exc).__name__, str(exc))
            attrs = []
    else:
        attrs = []
    return itertools.chain(super(GetAttThenSelect, self).dep_attrs(resource_name), attrs)