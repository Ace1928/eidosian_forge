from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def get_blueprint_plan_flag():
    return base.Argument('--blueprint-plan-file', required=True, help='Path of the JSON file containing the blueprint plan.')