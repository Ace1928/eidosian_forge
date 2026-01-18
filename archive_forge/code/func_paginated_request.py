import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def paginated_request(self, *args, **kwargs):
    """
        Generic function to create a paginated request to any API call
        not only aggregated or zone ones as request_aggregated_items.

        @inherits: :class:`GoogleBaseConnection.request`
        """
    more_results = True
    items = []
    max_results = kwargs['max_results'] if 'max_results' in kwargs else 500
    params = {'maxResults': max_results}
    while more_results:
        self.gce_params = params
        response = self.request(*args, **kwargs)
        items.extend(response.object.get('items', []))
        more_results = 'pageToken' in params
    return {'items': items}