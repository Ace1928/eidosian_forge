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
def associate_aggregates(self, resource_provider_uuid, aggregates):
    """Associate a list of aggregates with a resource provider.

        :param resource_provider_uuid: UUID of the resource provider.
        :param aggregates: aggregates to be associated to the resource
                           provider.
        :returns: All aggregates associated with the resource provider.
        """
    url = '/resource_providers/%s/aggregates' % resource_provider_uuid
    return self._put(url, aggregates).json()