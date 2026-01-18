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
def ex_destroy_instancetemplate(self, instancetemplate):
    """
        Deletes the specified instance template. If you delete an instance
        template that is being referenced from another instance group, the
        instance group will not be able to create or recreate virtual machine
        instances. Deleting an instance template is permanent and cannot be
        undone.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute

        :param  instancetemplate:  The name of the instance template to
                                   delete.
        :type   instancetemplate: ``str``

        :return  instanceTemplate:  Return True if successful.
        :rtype   instanceTemplate: ````bool````
        """
    request = '/global/instanceTemplates/%s' % instancetemplate.name
    request_data = {}
    self.connection.async_request(request, method='DELETE', data=request_data)
    return True