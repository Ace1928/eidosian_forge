import functools
import re
import time
from urllib import parse
import uuid
import requests
from keystoneauth1 import exceptions as ks_exc
from keystoneauth1 import loading as keystone
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import versionutils
from neutron_lib._i18n import _
from neutron_lib.exceptions import placement as n_exc
@_check_placement_api_available
def list_allocations(self, consumer_uuid):
    """List allocations for the consumer

        :param consumer_uuid: The uuid of the consumer, in case of bound port
                              owned by a VM, the VM uuid.
        :returns: All allocation records for the consumer.
        """
    url = '/allocations/%s' % consumer_uuid
    return self._get(url).json()