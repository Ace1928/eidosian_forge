import re
import uuid
import google.auth
from google.auth import downscoped
from google.auth.transport import requests
from google.cloud import exceptions
from google.cloud import storage
from google.oauth2 import credentials
import pytest
def get_token_from_broker(bucket_name, object_prefix):
    """Simulates token broker generating downscoped tokens for specified bucket.

    Args:
        bucket_name (str): The name of the Cloud Storage bucket.
        object_prefix (str): The prefix string of the object name. This is used
            to ensure access is restricted to only objects starting with this
            prefix string.

    Returns:
        Tuple[str, datetime.datetime]: The downscoped access token and its expiry date.
    """
    available_resource = '//storage.googleapis.com/projects/_/buckets/{0}'.format(bucket_name)
    available_permissions = ['inRole:roles/storage.objectViewer']
    availability_expression = "resource.name.startsWith('projects/_/buckets/{0}/objects/{1}')".format(bucket_name, object_prefix)
    availability_condition = downscoped.AvailabilityCondition(availability_expression)
    rule = downscoped.AccessBoundaryRule(available_resource=available_resource, available_permissions=available_permissions, availability_condition=availability_condition)
    credential_access_boundary = downscoped.CredentialAccessBoundary(rules=[rule])
    source_credentials, _ = google.auth.default()
    if source_credentials.requires_scopes:
        source_credentials = source_credentials.with_scopes(['https://www.googleapis.com/auth/cloud-platform'])
    downscoped_credentials = downscoped.Credentials(source_credentials=source_credentials, credential_access_boundary=credential_access_boundary)
    downscoped_credentials.refresh(requests.Request())
    access_token = downscoped_credentials.token
    expiry = downscoped_credentials.expiry
    return (access_token, expiry)