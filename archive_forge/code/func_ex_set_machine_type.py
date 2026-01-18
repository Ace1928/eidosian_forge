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
def ex_set_machine_type(self, node, machine_type='n1-standard-1'):
    """
        Set the machine type of the stopped instance. Can be the short-name,
        a full, or partial URL.

        :param  node: Target node object to change
        :type   node: :class:`Node`

        :param  machine_type: Desired machine type
        :type   machine_type: ``str``

        :return:  True if successful
        :rtype:   ``bool``
        """
    request = mt_url = '/zones/%s' % node.extra['zone'].name
    mt = machine_type.split('/')[-1]
    mt_url = '{}/machineTypes/{}'.format(mt_url, mt)
    request = '{}/instances/{}/setMachineType'.format(request, node.name)
    body = {'machineType': mt_url}
    self.connection.async_request(request, method='POST', data=body)
    return True