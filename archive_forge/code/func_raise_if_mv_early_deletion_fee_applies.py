from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import errors as api_errors
from googlecloudsdk.command_lib.storage import errors as command_errors
from googlecloudsdk.command_lib.storage import manifest_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def raise_if_mv_early_deletion_fee_applies(object_resource):
    """Raises error if Google Cloud Storage object will incur an extra charge."""
    if not (properties.VALUES.storage.check_mv_early_deletion_fee.GetBool() and object_resource.storage_url.scheme is storage_url.ProviderPrefix.GCS and object_resource.creation_time and (object_resource.storage_class in _EARLY_DELETION_MINIMUM_DAYS)):
        return
    minimum_lifetime = _EARLY_DELETION_MINIMUM_DAYS[object_resource.storage_class.lower()]
    creation_datetime_utc = resource_util.convert_datetime_object_to_utc(object_resource.creation_time)
    current_datetime_utc = resource_util.convert_datetime_object_to_utc(datetime.datetime.now())
    if current_datetime_utc < creation_datetime_utc + datetime.timedelta(days=minimum_lifetime):
        raise exceptions.Error('Deleting {} may incur an early deletion charge. Note: the source object of a mv operation is deleted.\nThe object appears to have been created on {}, and the minimum time before deletion for the {} storage class is {} days.\nTo allow deleting the object anyways, run "gcloud config set storage/check_mv_early_deletion_fee False"'.format(object_resource, object_resource.creation_time, object_resource.storage_class, minimum_lifetime))