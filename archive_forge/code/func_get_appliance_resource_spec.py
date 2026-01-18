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
def get_appliance_resource_spec(resource_name='appliance'):
    return concepts.ResourceSpec(APPLIANCES_COLLECTION, resource_name=resource_name, appliancesId=appliance_attribute_config(name=resource_name), locationsId=region_attribute_config(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, disable_auto_completers=False)