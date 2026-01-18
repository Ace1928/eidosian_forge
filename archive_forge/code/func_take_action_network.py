import argparse
from cliff import columns as cliff_columns
from osc_lib.command import command
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
from openstackclient.network import utils as network_utils
def take_action_network(self, client, parsed_args):
    obj = client.find_security_group(parsed_args.group, ignore_missing=False)
    display_columns, property_columns = _get_columns(obj)
    data = utils.get_item_properties(obj, property_columns, formatters=_formatters_network)
    return (display_columns, data)