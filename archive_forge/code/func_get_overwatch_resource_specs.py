from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def get_overwatch_resource_specs():
    return concepts.ResourceSpec('securedlandingzone.organizations.locations.overwatches', resource_name='overwatch', organizationsId=organization_attribute_config(), locationsId=location_attribute_config(), overwatchesId=overwatch_attribute_config())