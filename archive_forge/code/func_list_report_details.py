from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.api_lib.storage.gcs_json import client as gcs_json_client
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import properties
def list_report_details(self, report_config_name, page_size=None):
    """Lists the report details."""
    return list_pager.YieldFromList(self.client.projects_locations_reportConfigs_reportDetails, self.messages.StorageinsightsProjectsLocationsReportConfigsReportDetailsListRequest(parent=report_config_name), batch_size=page_size if page_size is not None else PAGE_SIZE, batch_size_attribute='pageSize', field='reportDetails')