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
def add_list_resource_args(parser, listing_orders=True):
    """Add both order and appliance resource arguments for list commands.

  Args:
    parser (arg_parse.Parser): The parser for the command.
    listing_orders (bool): Toggles the help text phrasing to match either orders
      or appliances being the resource being listed.
  """
    verb = ResourceVerb.LIST
    primary_help = 'The {} to {}.'
    secondary_help = 'The {} associated with the {} to {}.'
    if listing_orders:
        orders_help = primary_help.format('orders', verb.value)
        appliances_help = secondary_help.format('appliances', 'orders', verb.value)
        parser.display_info.AddUriFunc(_get_order_uri)
    else:
        appliances_help = primary_help.format('appliances', verb.value)
        orders_help = secondary_help.format('orders', 'appliances', verb.value)
        parser.display_info.AddUriFunc(_get_appliance_uri)
    arg_specs = [presentation_specs.ResourcePresentationSpec('--appliances', get_appliance_resource_spec('appliances'), appliances_help, flag_name_overrides={'region': ''}, plural=True, prefixes=False), presentation_specs.ResourcePresentationSpec('--orders', get_order_resource_spec('orders'), orders_help, flag_name_overrides={'region': ''}, plural=True, prefixes=True)]
    concept_parsers.ConceptParser(arg_specs).AddToParser(parser)
    _add_region_flag(parser, verb)