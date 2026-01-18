import copy
import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
def convert_entries_to_gateway(entries):
    changed_entries = copy.deepcopy(entries)
    for entry in changed_entries:
        if 'nexthop' in entry:
            entry['gateway'] = entry['nexthop']
            del entry['nexthop']
    return changed_entries