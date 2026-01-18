from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.api_lib.storage.gcs_json import client as gcs_json_client
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import properties
def create_inventory_report(self, source_bucket, destination_url, metadata_fields=None, start_date=None, end_date=None, frequency=None, csv_separator=None, csv_delimiter=None, csv_header=None, parquet=None, display_name=None):
    """Creates a report config.

    Args:
      source_bucket (str): Source bucket name for which reports will be
        generated.
      destination_url (storage_url.CloudUrl): The destination url where the
        generated reports will be stored.
      metadata_fields (list[str]): Fields to be included in the report.
      start_date (datetime.datetime.date): The date to start generating reports.
      end_date (datetime.datetime.date): The date after which to stop generating
        reports.
      frequency (str): Can be either DAILY or WEEKLY.
      csv_separator (str): The character used to separate the records in the
        CSV file.
      csv_delimiter (str): The delimiter that separates the fields in the CSV
        file.
      csv_header (bool): If True, include the headers in the CSV file.
      parquet (bool): If True, set the parquet options.
      display_name (str): Display name for the report config.

    Returns:
      The created ReportConfig object.
    """
    frequency_options = self.messages.FrequencyOptions(startDate=self.messages.Date(year=start_date.year, month=start_date.month, day=start_date.day), endDate=self.messages.Date(year=end_date.year, month=end_date.month, day=end_date.day), frequency=getattr(self.messages.FrequencyOptions.FrequencyValueValuesEnum, frequency.upper()))
    object_metadata_report_options = self.messages.ObjectMetadataReportOptions(metadataFields=metadata_fields, storageDestinationOptions=self.messages.CloudStorageDestinationOptions(bucket=destination_url.bucket_name, destinationPath=destination_url.object_name), storageFilters=self.messages.CloudStorageFilters(bucket=source_bucket))
    report_format_options = self._get_report_format_options(csv_separator, csv_delimiter, csv_header, parquet)
    report_config = self.messages.ReportConfig(csvOptions=report_format_options.csv, parquetOptions=report_format_options.parquet, displayName=display_name, frequencyOptions=frequency_options, objectMetadataReportOptions=object_metadata_report_options)
    create_request = self.messages.StorageinsightsProjectsLocationsReportConfigsCreateRequest(parent=_get_parent_string_from_bucket(source_bucket), reportConfig=report_config)
    return self.client.projects_locations_reportConfigs.Create(create_request)