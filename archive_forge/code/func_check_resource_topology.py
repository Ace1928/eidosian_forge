import logging
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
def check_resource_topology(self, client, parsed_args):
    obj = client.validate_auto_allocated_topology(parsed_args.project)
    columns = _format_check_resource_columns()
    data = utils.get_item_properties(_format_check_resource(obj), columns, formatters={})
    return (columns, data)