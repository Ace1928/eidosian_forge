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
def get_port_binding_by_status_and_host(bindings, status, host='', raise_if_not_found=False, port_id=None):
    """Returns from an iterable the binding with the specified status and host.

    The input iterable can contain zero or one binding in status ACTIVE
    and zero or many bindings in status INACTIVE. As a consequence, to
    unequivocally retrieve an inactive binding, the caller must specify a non
    empty value for host. If host is the empty string, the first binding
    satisfying the specified status will be returned. If no binding is found
    with the specified status and host, None is returned or PortBindingNotFound
    is raised if raise_if_not_found is True

    :param bindings: An iterable containing port bindings
    :param status: The status of the port binding to return. Possible values
                   are ACTIVE or INACTIVE as defined in
                   :file:`neutron_lib/constants.py`.
    :param host: str representing the host of the binding to return.
    :param raise_if_not_found: If a binding is not found and this parameter is
                               True, a PortBindingNotFound exception is raised
    :param port_id: The id of the binding's port
    :returns: The searched for port binding or None if it is not found
    :raises: PortBindingNotFound if the binding is not found and
             raise_if_not_found is True
    """
    for binding in bindings:
        if binding[pb_ext.STATUS] == status:
            if not host or binding[pb_ext.HOST] == host:
                return binding
    if raise_if_not_found:
        raise exceptions.PortBindingNotFound(port_id=port_id, host=host)