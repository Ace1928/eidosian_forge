import copy
import glance_store as store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
import glance.api.common
import glance.common.exception as exception
from glance.common import utils
import glance.domain
import glance.domain.proxy
from glance.i18n import _, _LI
def added_new_properties(self):
    current_props = set(self.image.extra_properties.keys())
    return bool(current_props.difference(self.orig_props))