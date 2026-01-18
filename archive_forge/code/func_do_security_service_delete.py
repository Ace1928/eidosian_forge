from operator import xor
import os
import re
import sys
import time
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient.common.apiclient import utils as apiclient_utils
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
import manilaclient.v2.shares
@cliutils.arg('security_service', metavar='<security-service>', nargs='+', help='Name or ID of the security service(s) to delete.')
def do_security_service_delete(cs, args):
    """Delete one or more security services."""
    failure_count = 0
    for security_service in args.security_service:
        try:
            security_ref = _find_security_service(cs, security_service)
            cs.security_services.delete(security_ref)
        except Exception as e:
            failure_count += 1
            print('Delete for security service %s failed: %s' % (security_service, e), file=sys.stderr)
    if failure_count == len(args.security_service):
        raise exceptions.CommandError('Unable to delete any of the specified security services.')