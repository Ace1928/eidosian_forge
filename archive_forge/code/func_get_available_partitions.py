import copy
import os
import botocore.session
from botocore.client import Config
from botocore.exceptions import DataNotFoundError, UnknownServiceError
import boto3
import boto3.utils
from boto3.exceptions import ResourceNotExistsError, UnknownAPIVersionError
from .resources.factory import ResourceFactory
def get_available_partitions(self):
    """Lists the available partitions

        :rtype: list
        :return: Returns a list of partition names (e.g., ["aws", "aws-cn"])
        """
    return self._session.get_available_partitions()