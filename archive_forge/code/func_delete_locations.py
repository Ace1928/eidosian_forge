from oslo_config import cfg
from oslo_log import log as logging
import webob.exc
from glance.api import policy
from glance.common import exception
from glance.i18n import _
def delete_locations(self):
    self._enforce('delete_image_location')
    if not (CONF.oslo_policy.enforce_new_defaults or CONF.oslo_policy.enforce_scope):
        check_is_image_mutable(self._context, self._image)