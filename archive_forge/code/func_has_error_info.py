from googlecloudsdk.api_lib.util import exceptions as gcloud_exception
from googlecloudsdk.core import exceptions as gcloud_core_exceptions
def has_error_info(self, reason):
    return reason in [e['reason'] for e in self.error_info]