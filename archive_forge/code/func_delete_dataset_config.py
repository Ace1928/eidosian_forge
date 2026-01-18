from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.api_lib.storage.gcs_json import client as gcs_json_client
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import properties
def delete_dataset_config(self, dataset_config_relative_name):
    """Deletes the dataset config."""
    request = self.messages.StorageinsightsProjectsLocationsDatasetConfigsDeleteRequest(name=dataset_config_relative_name)
    return self.client.projects_locations_datasetConfigs.Delete(request)