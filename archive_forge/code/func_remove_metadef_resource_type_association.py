from oslo_config import cfg
from oslo_log import log as logging
import webob.exc
from glance.api import policy
from glance.common import exception
from glance.i18n import _
def remove_metadef_resource_type_association(self):
    self._enforce('remove_metadef_resource_type_association')