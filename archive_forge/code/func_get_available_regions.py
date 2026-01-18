import copy
import os
import botocore.session
from botocore.client import Config
from botocore.exceptions import DataNotFoundError, UnknownServiceError
import boto3
import boto3.utils
from boto3.exceptions import ResourceNotExistsError, UnknownAPIVersionError
from .resources.factory import ResourceFactory
def get_available_regions(self, service_name, partition_name='aws', allow_non_regional=False):
    """Lists the region and endpoint names of a particular partition.

        :type service_name: string
        :param service_name: Name of a service to list endpoint for (e.g., s3).

        :type partition_name: string
        :param partition_name: Name of the partition to limit endpoints to.
            (e.g., aws for the public AWS endpoints, aws-cn for AWS China
            endpoints, aws-us-gov for AWS GovCloud (US) Endpoints, etc.)

        :type allow_non_regional: bool
        :param allow_non_regional: Set to True to include endpoints that are
             not regional endpoints (e.g., s3-external-1,
             fips-us-gov-west-1, etc).

        :return: Returns a list of endpoint names (e.g., ["us-east-1"]).
        """
    return self._session.get_available_regions(service_name=service_name, partition_name=partition_name, allow_non_regional=allow_non_regional)