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
def is_user(self):
    """Returns whether the credentials represent a user (True) or workload (False).
        Workloads behave similarly to service accounts. Currently workloads will use
        service account impersonation but will eventually not require impersonation.
        As a result, this property is more reliable than the service account email
        property in determining if the credentials represent a user or workload.

        Returns:
            bool: True if the credentials represent a user. False if they represent a
                workload.
        """
    if self._service_account_impersonation_url:
        return False
    return self.is_workforce_pool