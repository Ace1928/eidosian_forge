from the current environment without the need to copy, save and manage
import abc
import copy
import datetime
import io
import json
import re
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.auth import impersonated_credentials
from google.auth import metrics
from google.oauth2 import sts
from google.oauth2 import utils
@property
def service_account_email(self):
    """Returns the service account email if service account impersonation is used.

        Returns:
            Optional[str]: The service account email if impersonation is used. Otherwise
                None is returned.
        """
    if self._service_account_impersonation_url:
        url = self._service_account_impersonation_url
        start_index = url.rfind('/')
        end_index = url.find(':generateAccessToken')
        if start_index != -1 and end_index != -1 and (start_index < end_index):
            start_index = start_index + 1
            return url[start_index:end_index]
    return None