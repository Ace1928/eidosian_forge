import copy
import os_service_types.data
from os_service_types import exc
def get_official_service_data(self, service_type):
    """Get the service data for an official service_type.

        :param str service_type: The official service-type to get data for.
        :returns dict: Service data for the service or None if not found.
        """
    service_type = _normalize_type(service_type)
    for service in self._service_types_data['services']:
        if service_type == service['service_type']:
            return service
    return None