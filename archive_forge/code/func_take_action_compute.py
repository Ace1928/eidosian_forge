import argparse
from cliff import columns as cliff_columns
from osc_lib.command import command
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
from openstackclient.network import utils as network_utils
def take_action_compute(self, client, parsed_args):
    obj = client.api.security_group_find(parsed_args.group)
    display_columns, property_columns = _get_columns(obj)
    data = utils.get_dict_properties(obj, property_columns, formatters=_formatters_compute)
    return (display_columns, data)