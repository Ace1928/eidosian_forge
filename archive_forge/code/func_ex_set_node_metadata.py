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
def ex_set_node_metadata(self, node, metadata):
    """
        Set metadata for the specified node.

        :keyword  node: The existing target Node (instance) for the request.
        :type     node: ``Node``

        :keyword  metadata: Set (or clear with None) metadata for this
                            particular node.
        :type     metadata: ``dict`` or ``None``

        :return: True if successful
        :rtype:  ``bool``
        """
    if not isinstance(node, Node):
        raise ValueError('Must specify a valid libcloud node object.')
    node_name = node.name
    zone_name = node.extra['zone'].name
    if 'metadata' in node.extra and 'fingerprint' in node.extra['metadata']:
        current_fp = node.extra['metadata']['fingerprint']
    else:
        current_fp = 'absent'
    body = self._format_metadata(current_fp, metadata)
    request = '/zones/{}/instances/{}/setMetadata'.format(zone_name, node_name)
    self.connection.async_request(request, method='POST', data=body)
    return True