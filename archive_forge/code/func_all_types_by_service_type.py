import copy
import os_service_types.data
from os_service_types import exc
@property
def all_types_by_service_type(self):
    """Mapping of official service type to official type and aliases."""
    return copy.deepcopy(self._service_types_data['all_types_by_service_type'])