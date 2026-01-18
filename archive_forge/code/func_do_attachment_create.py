import argparse
import collections
import os
from oslo_utils import strutils
import cinderclient
from cinderclient import api_versions
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3.shell_base import *  # noqa
from cinderclient.v3.shell_base import CheckSizeArgForCreate
@api_versions.wraps('3.27')
@utils.arg('volume', metavar='<volume>', help='Name or ID of volume or volumes to attach.')
@utils.arg('server_id', metavar='<server_id>', nargs='?', default=None, help='ID of server attaching to.')
@utils.arg('--connect', metavar='<connect>', default=False, help='Make an active connection using provided connector info (True or False).')
@utils.arg('--initiator', metavar='<initiator>', default=None, help='iqn of the initiator attaching to.  Default=None.')
@utils.arg('--ip', metavar='<ip>', default=None, help='ip of the system attaching to.  Default=None.')
@utils.arg('--host', metavar='<host>', default=None, help='Name of the host attaching to. Default=None.')
@utils.arg('--platform', metavar='<platform>', default='x86_64', help='Platform type. Default=x86_64.')
@utils.arg('--ostype', metavar='<ostype>', default='linux2', help='OS type. Default=linux2.')
@utils.arg('--multipath', metavar='<multipath>', default=False, help='Use multipath. Default=False.')
@utils.arg('--mountpoint', metavar='<mountpoint>', default=None, help='Mountpoint volume will be attached at. Default=None.')
@utils.arg('--mode', metavar='<mode>', default='null', start_version='3.54', help='Mode of attachment, rw, ro and null, where null indicates we want to honor any existing admin-metadata settings.  Default=null.')
def do_attachment_create(cs, args):
    """Create an attachment for a cinder volume."""
    connector = {}
    if strutils.bool_from_string(args.connect, strict=True):
        connector = {'initiator': args.initiator, 'ip': args.ip, 'platform': args.platform, 'host': args.host, 'os_type': args.ostype, 'multipath': args.multipath, 'mountpoint': args.mountpoint}
    volume = utils.find_volume(cs, args.volume)
    mode = getattr(args, 'mode', 'null')
    attachment = cs.attachments.create(volume.id, connector, args.server_id, mode)
    connector_dict = attachment.pop('connection_info', None)
    shell_utils.print_dict(attachment)
    if connector_dict:
        shell_utils.print_dict(connector_dict)