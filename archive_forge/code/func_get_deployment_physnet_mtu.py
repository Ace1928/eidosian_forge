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
def get_deployment_physnet_mtu():
    """Retrieves global physical network MTU setting.

    Plugins should use this function to retrieve the MTU set by the operator
    that is equal to or less than the MTU of their nodes' physical interfaces.
    Note that it is the responsibility of the plugin to deduct the value of
    any encapsulation overhead required before advertising it to VMs.

    Note that this function depends on the global_physnet_mtu config option
    being registered in the global CONF.

    :returns: The global_physnet_mtu from the global CONF.
    """
    return cfg.CONF.global_physnet_mtu