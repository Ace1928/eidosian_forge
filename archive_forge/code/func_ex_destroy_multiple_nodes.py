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
def ex_destroy_multiple_nodes(self, node_list, ignore_errors=True, destroy_boot_disk=False, poll_interval=2, timeout=DEFAULT_TASK_COMPLETION_TIMEOUT):
    """
        Destroy multiple nodes at once.

        :param  node_list: List of nodes to destroy
        :type   node_list: ``list`` of :class:`Node`

        :keyword  ignore_errors: If true, don't raise an exception if one or
                                 more nodes fails to be destroyed.
        :type     ignore_errors: ``bool``

        :keyword  destroy_boot_disk: If true, also destroy the nodes' boot
                                     disks.
        :type     destroy_boot_disk: ``bool``

        :keyword  poll_interval: Number of seconds between status checks.
        :type     poll_interval: ``int``

        :keyword  timeout: Number of seconds to wait for all nodes to be
                           destroyed.
        :type     timeout: ``int``

        :return:  A list of boolean values.  One for each node.  True means
                  that the node was successfully destroyed.
        :rtype:   ``list`` of ``bool``
        """
    status_list = []
    complete = False
    start_time = time.time()
    for node in node_list:
        request = '/zones/{}/instances/{}'.format(node.extra['zone'].name, node.name)
        try:
            response = self.connection.request(request, method='DELETE').object
        except GoogleBaseError:
            self._catch_error(ignore_errors=ignore_errors)
            response = None
        status = {'node': node, 'node_success': False, 'node_response': response, 'disk_success': not destroy_boot_disk, 'disk_response': None}
        status_list.append(status)
    while not complete:
        if time.time() - start_time >= timeout:
            raise Exception('Timeout (%s sec) while waiting to delete multiple instances')
        complete = True
        for status in status_list:
            operation = status['node_response'] or status['disk_response']
            delete_disk = False
            if operation:
                no_errors = True
                try:
                    response = self.connection.request(operation['selfLink']).object
                except GoogleBaseError:
                    self._catch_error(ignore_errors=ignore_errors)
                    no_errors = False
                    response = {'status': 'DONE'}
                if response['status'] == 'DONE':
                    if status['node_response']:
                        status['node_response'] = None
                        status['node_success'] = no_errors
                        delete_disk = True
                    else:
                        status['disk_response'] = None
                        status['disk_success'] = no_errors
            if delete_disk and destroy_boot_disk:
                boot_disk = status['node'].extra['boot_disk']
                if boot_disk:
                    request = '/zones/{}/disks/{}'.format(boot_disk.extra['zone'].name, boot_disk.name)
                    try:
                        response = self.connection.request(request, method='DELETE').object
                    except GoogleBaseError:
                        self._catch_error(ignore_errors=ignore_errors)
                        no_errors = False
                        response = None
                    status['disk_response'] = response
                else:
                    status['disk_success'] = True
            operation = status['node_response'] or status['disk_response']
            if operation:
                time.sleep(poll_interval)
                complete = False
    success = []
    for status in status_list:
        s = status['node_success'] and status['disk_success']
        success.append(s)
    return success