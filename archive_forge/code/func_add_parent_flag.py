from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def add_parent_flag(parser):
    concept_parsers.ConceptParser.ForResource('PARENT', get_parent_resource_specs(), 'Parent of the overwatch instances.', required=True).AddToParser(parser)