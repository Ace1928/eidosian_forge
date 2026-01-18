from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def add_operation_flag(parser):
    concept_parsers.ConceptParser.ForResource('OPERATION', get_operation_resource_specs(), 'Name of the longrunning operation.', required=True).AddToParser(parser)