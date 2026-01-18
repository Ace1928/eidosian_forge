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
def ensure_resource_provider(self, resource_provider):
    """Ensure a resource provider exists by updating or creating it.

        :param resource_provider: The resource provider. A dict with
                                  the uuid (required),
                                  the name (required) and
                                  the parent_provider_uuid (optional).
        :returns: The Resource Provider updated or created.

        Beware, this is not an atomic operation of the API.
        """
    try:
        return self.update_resource_provider(resource_provider=resource_provider)
    except n_exc.PlacementResourceProviderNotFound:
        return self.create_resource_provider(resource_provider)