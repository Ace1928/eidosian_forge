from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.api_lib.storage.gcs_json import client as gcs_json_client
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import properties
def delete_inventory_report(self, report_config_name, force=False):
    """Deletes the report config."""
    request = self.messages.StorageinsightsProjectsLocationsReportConfigsDeleteRequest(name=report_config_name, force=force)
    return self.client.projects_locations_reportConfigs.Delete(request)