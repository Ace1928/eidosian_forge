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
def ex_instancegroupmanager_list_managed_instances(self, manager):
    """
        Lists all of the instances in the Managed Instance Group.

        Each instance in the list has a currentAction, which indicates
        the action that the managed instance group is performing on the
        instance. For example, if the group is still creating an instance,
        the currentAction is 'CREATING'.  Note that 'instanceStatus' might not
        be available, for example, if currentAction is 'CREATING' or
        'RECREATING'. If a previous action failed, the list displays the errors
        for that failed action.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute
        * https://www.googleapis.com/auth/compute.readonly

        'currentAction' values are one of:
           'ABANDONING', 'CREATING', 'DELETING', 'NONE',
           'RECREATING', 'REFRESHING', 'RESTARTING'

        :param  manager: Instance Group Manager to operate on.
        :type   manager: :class:`GCEInstanceGroupManager`

        :return: ``list`` of ``dict`` containing 'name', 'zone', 'lastAttempt',
                 'currentAction', 'instance' and 'instanceStatus'.
        :rtype: ``list``
        """
    request = '/zones/{}/instanceGroupManagers/{}/listManagedInstances'.format(manager.zone.name, manager.name)
    response = self.connection.request(request, method='POST').object
    instance_data = []
    if 'managedInstances' in response:
        for i in response['managedInstances']:
            i['name'] = self._get_components_from_path(i['instance'])['name']
            i['zone'] = manager.zone.name
            instance_data.append(i)
    return instance_data