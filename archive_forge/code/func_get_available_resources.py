import copy
import os
import botocore.session
from botocore.client import Config
from botocore.exceptions import DataNotFoundError, UnknownServiceError
import boto3
import boto3.utils
from boto3.exceptions import ResourceNotExistsError, UnknownAPIVersionError
from .resources.factory import ResourceFactory
def get_available_resources(self):
    """
        Get a list of available services that can be loaded as resource
        clients via :py:meth:`Session.resource`.

        :rtype: list
        :return: List of service names
        """
    return self._loader.list_available_services(type_name='resources-1')