from googlecloudsdk.api_lib.util import exceptions as gcloud_exception
from googlecloudsdk.core import exceptions as gcloud_core_exceptions
@property
def error_info(self):
    return self.http_exception.payload.type_details['ErrorInfo']