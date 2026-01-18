import argparse
import collections
import datetime
import getpass
import logging
import os
import pprint
import sys
import time
from oslo_utils import netutils
from oslo_utils import strutils
from oslo_utils import timeutils
import novaclient
from novaclient import api_versions
from novaclient import base
from novaclient import client
from novaclient import exceptions
from novaclient.i18n import _
from novaclient import shell
from novaclient import utils
from novaclient.v2 import availability_zones
from novaclient.v2 import quotas
from novaclient.v2 import servers
@utils.arg('server', metavar='<server>', help=_('Name or ID of server.'))
@utils.arg('--port', dest='port', action='store', type=int, default=22, help=_('Optional flag to indicate which port to use for ssh. (Default=22)'))
@utils.arg('--private', dest='private', action='store_true', default=False, help=argparse.SUPPRESS)
@utils.arg('--address-type', dest='address_type', action='store', type=str, default='floating', help=_('Optional flag to indicate which IP type to use. Possible values  includes fixed and floating (the Default).'))
@utils.arg('--network', metavar='<network>', help=_('Network to use for the ssh.'), default=None)
@utils.arg('--ipv6', dest='ipv6', action='store_true', default=False, help=_('Optional flag to indicate whether to use an IPv6 address attached to a server. (Defaults to IPv4 address)'))
@utils.arg('--login', metavar='<login>', help=_('Login to use.'), default='root')
@utils.arg('-i', '--identity', dest='identity', help=_('Private key file, same as the -i option to the ssh command.'), default='')
@utils.arg('--extra-opts', dest='extra', help=_('Extra options to pass to ssh. see: man ssh.'), default='')
def do_ssh(cs, args):
    """SSH into a server."""
    if '@' in args.server:
        user, server = args.server.split('@', 1)
        args.login = user
        args.server = server
    addresses = _find_server(cs, args.server).addresses
    address_type = 'fixed' if args.private else args.address_type
    version = 6 if args.ipv6 else 4
    pretty_version = 'IPv%d' % version
    if args.network:
        network_addresses = addresses.get(args.network)
        if not network_addresses:
            msg = _("Server '%(server)s' is not attached to network '%(network)s'")
            raise exceptions.ResourceNotFound(msg % {'server': args.server, 'network': args.network})
    elif len(addresses) > 1:
        msg = _("Server '%(server)s' is attached to more than one network. Please pick the network to use.")
        raise exceptions.CommandError(msg % {'server': args.server})
    elif not addresses:
        msg = _("Server '%(server)s' is not attached to any network.")
        raise exceptions.CommandError(msg % {'server': args.server})
    else:
        network_addresses = list(addresses.values())[0]
    match = lambda addr: all((addr.get('version') == version, addr.get('OS-EXT-IPS:type', 'floating') == address_type))
    matching_addresses = [address.get('addr') for address in network_addresses if match(address)]
    if not any(matching_addresses):
        msg = _("No address that would match network '%(network)s' and type '%(address_type)s' of version %(pretty_version)s has been found for server '%(server)s'.")
        raise exceptions.ResourceNotFound(msg % {'network': args.network, 'address_type': address_type, 'pretty_version': pretty_version, 'server': args.server})
    elif len(matching_addresses) > 1:
        msg = _('More than one %(pretty_version)s %(address_type)s address found.')
        raise exceptions.CommandError(msg % {'pretty_version': pretty_version, 'address_type': address_type})
    else:
        ip_address = matching_addresses[0]
    identity = '-i %s' % args.identity if len(args.identity) else ''
    cmd = 'ssh -%d -p%d %s %s@%s %s' % (version, args.port, identity, args.login, ip_address, args.extra)
    logger.debug("Executing cmd '%s'", cmd)
    os.system(cmd)