import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple
from ray.serve._private.common import (
from ray.serve._private.constants import (
from ray.serve.handle import RayServeHandle
def match_route(self, target_route: str) -> Optional[Tuple[str, RayServeHandle, bool]]:
    """Return the longest prefix match among existing routes for the route.
        Args:
            target_route: route to match against.
        Returns:
            (route, handle, is_cross_language) if found, else None.
        """
    for route in self.sorted_routes:
        if target_route.startswith(route):
            matched = False
            if route.endswith('/'):
                matched = True
            elif len(target_route) == len(route) or target_route[len(route)] == '/':
                matched = True
            if matched:
                endpoint = self.route_info[route]
                return (route, self.handles[endpoint], self.app_to_is_cross_language[endpoint.app])
    return None