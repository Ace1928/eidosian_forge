import logging
import os
from botocore import BOTOCORE_ROOT
from botocore.compat import HAS_GZIP, OrderedDict, json
from botocore.exceptions import DataNotFoundError, UnknownServiceError
from botocore.utils import deep_merge
@instance_cache
def list_api_versions(self, service_name, type_name):
    """List all API versions available for a particular service type

        :type service_name: str
        :param service_name: The name of the service

        :type type_name: str
        :param type_name: The type name for the service (i.e service-2,
            paginators-1, etc.)

        :rtype: list
        :return: A list of API version strings in sorted order.

        """
    known_api_versions = set()
    for possible_path in self._potential_locations(service_name, must_exist=True, is_dir=True):
        for dirname in os.listdir(possible_path):
            full_path = os.path.join(possible_path, dirname, type_name)
            if self.file_loader.exists(full_path):
                known_api_versions.add(dirname)
    if not known_api_versions:
        raise DataNotFoundError(data_path=service_name)
    return sorted(known_api_versions)