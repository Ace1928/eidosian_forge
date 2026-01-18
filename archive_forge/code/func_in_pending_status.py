import collections
import contextlib
import hashlib
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from webob import exc as web_exc
from neutron_lib._i18n import _
from neutron_lib.api import attributes
from neutron_lib.api.definitions import network as net_apidef
from neutron_lib.api.definitions import port as port_apidef
from neutron_lib.api.definitions import portbindings as pb
from neutron_lib.api.definitions import portbindings_extended as pb_ext
from neutron_lib.api.definitions import subnet as subnet_apidef
from neutron_lib import constants
from neutron_lib import exceptions
def in_pending_status(status):
    """Return True if status is a form of pending"""
    return status in (constants.PENDING_CREATE, constants.PENDING_UPDATE, constants.PENDING_DELETE)