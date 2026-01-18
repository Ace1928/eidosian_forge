import json
from libcloud.compute.base import (
from libcloud.common.gig_g8 import G8Connection
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def list_sizes(self):
    """
        Returns a list of node sizes as a cloud provider might have

        """
    location = self._location['locationCode']
    sizes = []
    for size in self._api_request('/sizes/list', {'location': location}):
        sizes.extend(self._to_size(size))
    return sizes