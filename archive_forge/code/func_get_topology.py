import logging
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
def get_topology(self, client, parsed_args):
    obj = client.get_auto_allocated_topology(parsed_args.project)
    display_columns, columns = _get_columns(obj)
    data = utils.get_item_properties(obj, columns, formatters={})
    return (display_columns, data)