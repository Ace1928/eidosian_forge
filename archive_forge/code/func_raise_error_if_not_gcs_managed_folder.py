from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
def raise_error_if_not_gcs_managed_folder(command_list, url):
    raise_error_if_not_gcs(command_list, url, example='gs://bucket/folder/')
    if not (isinstance(url, storage_url.CloudUrl) and url.is_object()):
        _raise_error_for_wrong_resource_type(command_list, 'Google Cloud Storage managed folder', 'gs://bucket/folder/', url)