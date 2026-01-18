import errno
from eventlet.green import socket
import functools
import os
import re
import urllib
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import netutils
from oslo_utils import strutils
from webob import exc
from glance.common import exception
from glance.common import location_strategy
from glance.common import timeutils
from glance.common import wsgi
from glance.i18n import _, _LE, _LW
def parse_valid_host_port(host_port):
    """
    Given a "host:port" string, attempts to parse it as intelligently as
    possible to determine if it is valid. This includes IPv6 [host]:port form,
    IPv4 ip:port form, and hostname:port or fqdn:port form.

    Invalid inputs will raise a ValueError, while valid inputs will return
    a (host, port) tuple where the port will always be of type int.
    """
    try:
        try:
            host, port = netutils.parse_host_port(host_port)
        except Exception:
            raise ValueError(_('Host and port "%s" is not valid.') % host_port)
        if not netutils.is_valid_port(port):
            raise ValueError(_('Port "%s" is not valid.') % port)
        if not (netutils.is_valid_ipv6(host) or netutils.is_valid_ipv4(host) or is_valid_hostname(host) or is_valid_fqdn(host)):
            raise ValueError(_('Host "%s" is not valid.') % host)
    except Exception as ex:
        raise ValueError(_('%s Please specify a host:port pair, where host is an IPv4 address, IPv6 address, hostname, or FQDN. If using an IPv6 address, enclose it in brackets separately from the port (i.e., "[fe80::a:b:c]:9876").') % ex)
    return (host, int(port))