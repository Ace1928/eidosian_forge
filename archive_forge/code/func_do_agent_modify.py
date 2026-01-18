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
@utils.arg('id', metavar='<id>', help=_('ID of the agent-build.'))
@utils.arg('version', metavar='<version>', help=_('Version.'))
@utils.arg('url', metavar='<url>', help=_('URL'))
@utils.arg('md5hash', metavar='<md5hash>', help=_('MD5 hash.'))
def do_agent_modify(cs, args):
    """DEPRECATED Modify existing agent build."""
    _emit_agent_deprecation_warning()
    result = cs.agents.update(args.id, args.version, args.url, args.md5hash)
    utils.print_dict(result.to_dict())