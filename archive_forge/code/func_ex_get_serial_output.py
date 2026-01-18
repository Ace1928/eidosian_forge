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
def ex_get_serial_output(self, node):
    """
        Fetch the console/serial port output from the node.

        :keyword  node: The existing target Node (instance) for the request.
        :type     node: ``Node``

        :return: A string containing serial port output of the node.
        :rtype:  ``str``
        """
    if not isinstance(node, Node):
        raise ValueError('Must specify a valid libcloud node object.')
    node_name = node.name
    zone_name = node.extra['zone'].name
    request = '/zones/{}/instances/{}/serialPort'.format(zone_name, node_name)
    response = self.connection.request(request, method='GET').object
    return response['contents']