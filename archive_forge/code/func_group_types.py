import typing as ty
from openstack.block_storage import _base_proxy
from openstack.block_storage.v3 import attachment as _attachment
from openstack.block_storage.v3 import availability_zone
from openstack.block_storage.v3 import backup as _backup
from openstack.block_storage.v3 import block_storage_summary as _summary
from openstack.block_storage.v3 import capabilities as _capabilities
from openstack.block_storage.v3 import extension as _extension
from openstack.block_storage.v3 import group as _group
from openstack.block_storage.v3 import group_snapshot as _group_snapshot
from openstack.block_storage.v3 import group_type as _group_type
from openstack.block_storage.v3 import limits as _limits
from openstack.block_storage.v3 import quota_set as _quota_set
from openstack.block_storage.v3 import resource_filter as _resource_filter
from openstack.block_storage.v3 import service as _service
from openstack.block_storage.v3 import snapshot as _snapshot
from openstack.block_storage.v3 import stats as _stats
from openstack.block_storage.v3 import transfer as _transfer
from openstack.block_storage.v3 import type as _type
from openstack.block_storage.v3 import volume as _volume
from openstack import exceptions
from openstack.identity.v3 import project as _project
from openstack import resource
from openstack import utils
def group_types(self, **query):
    """Retrive a generator of group types

        :param dict query: Optional query parameters to be sent to limit the
            resources being returned:

            * sort: Comma-separated list of sort keys and optional sort
              directions in the form of <key> [:<direction>]. A valid
              direction is asc (ascending) or desc (descending).
            * limit: Requests a page size of items. Returns a number of items
              up to a limit value. Use the limit parameter to make an
              initial limited request and use the ID of the last-seen item
              from the response as the marker parameter value in a
              subsequent limited request.
            * offset: Used in conjunction with limit to return a slice of
              items. Is where to start in the list.
            * marker: The ID of the last-seen item.

        :returns: A generator of group type objects.
        """
    return self._list(_group_type.GroupType, **query)