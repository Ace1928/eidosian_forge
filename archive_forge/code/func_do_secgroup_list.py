import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.service_type('database')
def do_secgroup_list(cs, args):
    """Lists all security groups."""
    items = cs.security_groups.list()
    sec_grps = items
    while items.next:
        items = cs.security_groups.list()
        sec_grps += items
    utils.print_list(sec_grps, ['id', 'name', 'instance_id'])