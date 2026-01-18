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
def appliance_attribute_config(name='appliance'):
    return concepts.ResourceParameterAttributeConfig(name=name, help_text='The appliance affiliated with the {resource}.', completion_request_params={'fieldMask': 'name'}, completion_id_field='name')