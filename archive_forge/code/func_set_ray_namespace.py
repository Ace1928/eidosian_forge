import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from ray.util.annotations import PublicAPI
def set_ray_namespace(self, ray_namespace: str) -> None:
    """Set Ray :ref:`namespace <namespaces-guide>`.

        Args:
            ray_namespace: The namespace to set.
        """
    if ray_namespace != self.ray_namespace:
        self.ray_namespace = ray_namespace
        self._cached_pb = None