import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('security_group_rule', metavar='<security_group_rule>', help=_('ID of security group rule.'))
@utils.service_type('database')
def do_secgroup_delete_rule(cs, args):
    """Deletes a security group rule."""
    cs.security_group_rules.delete(args.security_group_rule)