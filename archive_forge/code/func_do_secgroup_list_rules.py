import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('security_group', metavar='<security_group>', help=_('Security group ID.'))
@utils.service_type('database')
def do_secgroup_list_rules(cs, args):
    """Lists all rules for a security group."""
    sec_grp = cs.security_groups.get(args.security_group)
    rules = sec_grp._info['rules']
    utils.print_list(rules, ['id', 'protocol', 'from_port', 'to_port', 'cidr'], obj_is_dict=True)