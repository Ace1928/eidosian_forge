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
def delete_resource_provider_inventories(self, resource_provider_uuid):
    """Delete all inventory records for the resource provider.

        :param resource_provider_uuid: UUID of the resource provider.
        :raises PlacementResourceProviderNotFound: If the resource provider
                                                   is not found.
        :returns: None.
        """
    url = '/resource_providers/%s/inventories' % resource_provider_uuid
    try:
        self._delete(url)
    except ks_exc.NotFound as e:
        if 'No resource provider with uuid' in e.details:
            raise n_exc.PlacementResourceProviderNotFound(resource_provider=resource_provider_uuid)
        raise