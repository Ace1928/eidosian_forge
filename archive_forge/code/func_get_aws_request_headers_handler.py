from botocore.session import Session
from .aws_auth import AWSRequestsAuth
def get_aws_request_headers_handler(self, r):
    credentials = get_credentials(self._refreshable_credentials)
    return self.get_aws_request_headers(r, **credentials)