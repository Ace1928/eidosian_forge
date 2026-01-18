import copy
from docker.errors import create_unexpected_kwargs_error, InvalidArgument
from docker.types import TaskTemplate, ContainerSpec, Placement, ServiceMode
from .resource import Model, Collection
def force_update(self):
    """
        Force update the service even if no changes require it.

        Returns:
            bool: ``True`` if successful.
        """
    return self.update(force_update=True, fetch_current_spec=True)