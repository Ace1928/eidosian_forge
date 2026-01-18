import copy
import os_service_types.data
from os_service_types import exc
@property
def primary_service_by_project(self):
    """Mapping of project name to the primary associated service."""
    return copy.deepcopy(self._service_types_data['primary_service_by_project'])