import json
import datetime
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState, InvalidCredsError
import asyncio
def ex_describe_all_addresses_for_project(self, ex_project_id, include=None, only_associated=False):
    """
        Returns all the reserved IP addresses for this project
        optionally, returns only addresses associated with nodes.

        :param    only_associated: If true, return only the addresses
                                   that are associated with an instance.
        :type     only_associated: ``bool``

        :return:  List of IP addresses.
        :rtype:   ``list`` of :class:`dict`
        """
    path = '/metal/v1/projects/%s/ips' % ex_project_id
    params = {'include': include}
    ip_addresses = self.connection.request(path, params=params).object
    result = [a for a in ip_addresses.get('ip_addresses', []) if not only_associated or len(a.get('assignments', [])) > 0]
    return result