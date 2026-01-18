from oslo_config import cfg
from oslo_log import log as logging
import webob.exc
from glance.api import policy
from glance.common import exception
from glance.i18n import _
def add_metadef_property(self):
    self._enforce('add_metadef_property')