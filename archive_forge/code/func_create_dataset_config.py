from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.api_lib.storage.gcs_json import client as gcs_json_client
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import properties
def create_dataset_config(self, dataset_config_name, location, destination_project, source_projects_list, organization_number, retention_period, include_buckets_prefix_regex_list=None, exclude_buckets_prefix_regex_list=None, include_buckets_name_list=None, exclude_buckets_name_list=None, include_source_locations=None, exclude_source_locations=None, auto_add_new_buckets=False, identity_type=None, description=None):
    """Creates a dataset config.

    Args:
      dataset_config_name (str): Name for the dataset config being created.
      location (str): The location where insights data will be stored in a GCS
        managed BigQuery instance.
      destination_project (str): The project in which the dataset config is
        being created and by extension the insights data will be stored.
      source_projects_list (list[int]): List of source project numbers. Insights
        data is to be collected for buckets that belong to these projects.
      organization_number (int): Organization number of the organization to
        which all source projects must belong.
      retention_period (int): No of days for which insights data is to be
        retained in BigQuery instance.
      include_buckets_prefix_regex_list (list[str]): List of bucket prefix regex
        patterns which are to be included for insights processing from the
        source projects. We can either use included or excluded bucket
        parameters.
      exclude_buckets_prefix_regex_list (list[str]): List of bucket prefix regex
        patterns which are to be excluded from insights processing from the
        source projects. We can either use included or excluded bucket
        parameters.
      include_buckets_name_list (list[str]): List of bucket names which are to
        be included for insights processing from the source projects. We can
        either use included or excluded bucket parameters.
      exclude_buckets_name_list (list[str]): List of bucket names which are to
        be excluded from insights processing from the source projects. We can
        either use included or excluded bucket parameters.
      include_source_locations (list[str]): List of bucket locations which are
        to be included for insights processing from the source projects. We can
        either use included or excluded location parameters.
      exclude_source_locations (list[str]): List of bucket locations which are
        to be excluded from insights processing from the source projects. We can
        either use included or excluded location parameters.
      auto_add_new_buckets (bool): If True, auto includes any new buckets added
        to source projects that satisfy the include/exclude criterias.
      identity_type (str): Option for how permissions need to be setup for a
        given dataset config. Default option is IDENTITY_TYPE_PER_CONFIG.
      description (str): Human readable description text for the given dataset
        config.

    Returns:
      An instance of Operation message
    """
    if identity_type is not None:
        identity_type_enum = self.messages.Identity.TypeValueValuesEnum(identity_type.upper())
        identity_type = self.messages.Identity(type=identity_type_enum)
    else:
        identity_type = self.messages.Identity(type=self.messages.Identity.TypeValueValuesEnum.IDENTITY_TYPE_PER_CONFIG)
    source_projects = self.messages.SourceProjects(projectNumbers=source_projects_list)
    dataset_config = self.messages.DatasetConfig(description=description, identity=identity_type, includeNewlyCreatedBuckets=auto_add_new_buckets, name=dataset_config_name, organizationNumber=organization_number, retentionPeriodDays=retention_period, sourceProjects=source_projects)
    if exclude_buckets_name_list or exclude_buckets_prefix_regex_list:
        excluded_storage_buckets = [self.messages.CloudStorageBucket(bucketName=excluded_name) for excluded_name in exclude_buckets_name_list or []]
        excluded_storage_buckets += [self.messages.CloudStorageBucket(bucketPrefixRegex=excluded_regex) for excluded_regex in exclude_buckets_prefix_regex_list or []]
        dataset_config.excludeCloudStorageBuckets = self.messages.CloudStorageBuckets(cloudStorageBuckets=excluded_storage_buckets)
    if include_buckets_name_list or include_buckets_prefix_regex_list:
        included_storage_buckets = [self.messages.CloudStorageBucket(bucketName=included_name) for included_name in include_buckets_name_list or []]
        included_storage_buckets += [self.messages.CloudStorageBucket(bucketPrefixRegex=included_regex) for included_regex in include_buckets_prefix_regex_list or []]
        dataset_config.includeCloudStorageBuckets = self.messages.CloudStorageBuckets(cloudStorageBuckets=included_storage_buckets)
    if exclude_source_locations:
        dataset_config.excludeCloudStorageLocations = self.messages.CloudStorageLocations(locations=exclude_source_locations)
    if include_source_locations:
        dataset_config.includeCloudStorageLocations = self.messages.CloudStorageLocations(locations=include_source_locations)
    create_request = self.messages.StorageinsightsProjectsLocationsDatasetConfigsCreateRequest(datasetConfig=dataset_config, datasetConfigId=dataset_config_name, parent=_get_parent_string(destination_project, location))
    return self.client.projects_locations_datasetConfigs.Create(create_request)