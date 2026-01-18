from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.transfer.appliances import regions
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def parse_list_resource_args_as_filter_string(args, listing_orders=True):
    """Parses list resource args as a filter string.

  Args:
    args (parser_extensions.Namespace): the parsed arguments for the command.
    listing_orders (bool): Toggles the appropriate keys for order and appliance
      depending on which resource is primarily being listed.

  Returns:
    A filter string.
  """
    filter_list = [args.filter] if args.filter else []
    if args.IsSpecified('orders'):
        order_refs = args.CONCEPTS.orders.Parse()
        if order_refs:
            filter_key = 'name' if listing_orders else 'order'
            filter_list.append(_get_filter_clause_from_resources(filter_key, order_refs))
    if args.IsSpecified('appliances'):
        appliance_refs = args.CONCEPTS.appliances.Parse()
        if appliance_refs:
            filter_key = 'appliances' if listing_orders else 'name'
            filter_list.append(_get_filter_clause_from_resources(filter_key, appliance_refs))
    return ' AND '.join(filter_list)