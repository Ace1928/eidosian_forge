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
def is_valid_gre_id(gre_id):
    """Validate a GRE ID.

    :param gre_id: The GRE ID to validate.
    :returns: True if gre_id is a number that's a valid GRE ID.
    """
    return _is_valid_range(gre_id, constants.MIN_GRE_ID, constants.MAX_GRE_ID)