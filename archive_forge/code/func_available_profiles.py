import copy
import os
import botocore.session
from botocore.client import Config
from botocore.exceptions import DataNotFoundError, UnknownServiceError
import boto3
import boto3.utils
from boto3.exceptions import ResourceNotExistsError, UnknownAPIVersionError
from .resources.factory import ResourceFactory
@property
def available_profiles(self):
    """
        The profiles available to the session credentials
        """
    return self._session.available_profiles