from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.api_lib.storage.gcs_json import client as gcs_json_client
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import properties
def list_dataset_config(self, location=None, page_size=None):
    """Lists the dataset configs.

    Args:
      location (str): The location where insights data will be stored in a GCS
        managed BigQuery instance.
      page_size (int|None): Number of items per request to be returned.

    Returns:
      List of dataset configs.
    """
    if location is not None:
        parent = _get_parent_string(properties.VALUES.core.project.Get(), location)
    else:
        parent = _get_parent_string(properties.VALUES.core.project.Get(), '-')
    return list_pager.YieldFromList(self.client.projects_locations_datasetConfigs, self.messages.StorageinsightsProjectsLocationsDatasetConfigsListRequest(parent=parent), batch_size=page_size if page_size is not None else PAGE_SIZE, batch_size_attribute='pageSize', field='datasetConfigs')