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
def ex_create_instancegroupmanager(self, name, zone, template, size, base_instance_name=None, description=None):
    """
        Create a Managed Instance Group.

        :param  name: Name of the Instance Group.
        :type   name: ``str``

        :param  zone: The zone to which the Instance Group belongs
        :type   zone: ``str`` or :class:`GCEZone` or ``None``

        :param  template: The Instance Template.  Should be an instance
                                of GCEInstanceTemplate or a string.
        :type   template: ``str`` or :class:`GCEInstanceTemplate`

        :param  base_instance_name: The prefix for each instance created.
                                    If None, Instance Group name will be used.
        :type   base_instance_name: ``str``

        :param  description: User-supplied text about the Instance Group.
        :type   description: ``str``

        :return:  An Instance Group Manager object.
        :rtype:   :class:`GCEInstanceGroupManager`
        """
    zone = zone or self.zone
    if not hasattr(zone, 'name'):
        zone = self.ex_get_zone(zone)
    request = '/zones/%s/instanceGroupManagers' % zone.name
    manager_data = {}
    if not hasattr(template, 'name'):
        template = self.ex_get_instancetemplate(template)
    manager_data['instanceTemplate'] = template.extra['selfLink']
    manager_data['baseInstanceName'] = name
    if base_instance_name is not None:
        manager_data['baseInstanceName'] = base_instance_name
    manager_data['name'] = name
    manager_data['targetSize'] = size
    manager_data['description'] = description
    self.connection.async_request(request, method='POST', data=manager_data)
    return self.ex_get_instancegroupmanager(name, zone)