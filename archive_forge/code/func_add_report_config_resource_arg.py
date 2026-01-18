from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def add_report_config_resource_arg(parser, verb):
    """Adds a resource argument for storage insights report config.

  Args:
    parser: The argparse parser to add the resource arg to.
    verb: str, the verb to describe the resource, such as 'to update'.
  """
    concept_parsers.ConceptParser.ForResource('report_config', get_report_config_resource_spec(), 'The Report config {}.'.format(verb), required=True).AddToParser(parser)