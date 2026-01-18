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
def get_appliance_name(locations_id, appliances_id):
    """Returns an appliance name to locations and appliances ID."""
    return resources.Resource.RelativeName(resources.REGISTRY.Create(APPLIANCES_COLLECTION, appliancesId=appliances_id, locationsId=locations_id, projectsId=properties.VALUES.core.project.Get()))