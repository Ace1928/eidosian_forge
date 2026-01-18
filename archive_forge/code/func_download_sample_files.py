from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import textwrap
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.spanner import database_operations
from googlecloudsdk.api_lib.spanner import databases
from googlecloudsdk.api_lib.spanner import instances
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.spanner import ddl_parser
from googlecloudsdk.command_lib.spanner import samples
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
def download_sample_files(appname):
    """Download schema and binaries for the given sample app.

  If the schema and all binaries exist already, skip download. If any file
  doesn't exist, download them all.

  Args:
    appname: The name of the sample app, should exist in samples.APP_NAMES
  """
    storage_client = storage_api.StorageClient()
    bucket_ref = storage_util.BucketReference.FromUrl(samples.GCS_BUCKET)
    gcs_to_local = [(storage_util.ObjectReference.FromBucketRef(bucket_ref, samples.get_gcs_schema_name(appname)), samples.get_local_schema_path(appname))]
    gcs_bin_msgs = storage_client.ListBucket(bucket_ref, prefix=samples.get_gcs_bin_prefix(appname))
    bin_path = samples.get_local_bin_path(appname)
    for gcs_ref in gcs_bin_msgs:
        gcs_ref = storage_util.ObjectReference.FromMessage(gcs_ref)
        local_path = os.path.join(bin_path, gcs_ref.name.split('/')[-1])
        gcs_to_local.append((gcs_ref, local_path))
    if any((not os.path.exists(file_path) for _, file_path in gcs_to_local)):
        log.status.Print('Downloading files for the {} sample app'.format(appname))
        for gcs_ref, local_path in gcs_to_local:
            log.status.Print('Downloading {}'.format(local_path))
            local_dir = os.path.split(local_path)[0]
            if not os.path.exists(local_dir):
                files.MakeDir(local_dir)
            storage_client.CopyFileFromGCS(gcs_ref, local_path, overwrite=True)