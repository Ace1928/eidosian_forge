import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple
from ray.serve._private.common import (
from ray.serve._private.constants import (
from ray.serve.handle import RayServeHandle
def get_handle_for_endpoint(self, target_app_name: str) -> Optional[Tuple[str, RayServeHandle, bool]]:
    """Return the handle that matches with endpoint.

        Args:
            target_app_name: app_name to match against.
        Returns:
            (route, handle, app_name, is_cross_language) for the single app if there
            is only one, else find the app and handle for exact match. Else return None.
        """
    for endpoint_tag, handle in self.handles.items():
        if target_app_name == endpoint_tag.app or len(self.handles) == 1:
            endpoint_info = self.endpoints[endpoint_tag]
            return (endpoint_info.route, handle, endpoint_info.app_is_cross_language)
    return None