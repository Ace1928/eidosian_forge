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
def is_valid_vlan_tag(vlan):
    """Validate a VLAN tag.

    :param vlan: The VLAN tag to validate.
    :returns: True if vlan is a number that is a valid VLAN tag.
    """
    return _is_valid_range(vlan, constants.MIN_VLAN_TAG, constants.MAX_VLAN_TAG)