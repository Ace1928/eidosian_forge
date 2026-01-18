import logging
import os
from oslo_utils import strutils
from ironicclient.common import base
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import volume_connector
from ironicclient.v1 import volume_target
def list_volume_targets(self, node_id, marker=None, limit=None, sort_key=None, sort_dir=None, detail=False, fields=None, os_ironic_api_version=None, global_request_id=None):
    """List all the volume targets for a given node.

        :param node_id: Name or UUID of the node.
        :param marker: Optional, the UUID of a volume target, eg the last
                       volume target from a previous result set. Return
                       the next result set.
        :param limit: The maximum number of results to return per
                      request, if:

            1) limit > 0, the maximum number of volume targets to return.
            2) limit == 0, return the entire list of volume targets.
            3) limit param is NOT specified (None), the number of items
               returned respect the maximum imposed by the Ironic API
               (see Ironic's api.max_limit option).

        :param sort_key: Optional, field used for sorting.

        :param sort_dir: Optional, direction of sorting, either 'asc' (the
                         default) or 'desc'.

        :param detail: Optional, boolean whether to return detailed information
                       about volume targets.

        :param fields: Optional, a list with a specified set of fields
                       of the resource to be returned. Can not be used
                       when 'detail' is set.

        :param os_ironic_api_version: String version (e.g. "1.35") to use for
            the request.  If not specified, the client's default is used.

        :param global_request_id: String containing global request ID header
            value (in form "req-<UUID>") to use for the request.

        :returns: A list of volume targets.

        """
    if limit is not None:
        limit = int(limit)
    if detail and fields:
        raise exc.InvalidAttribute(_("Can't fetch a subset of fields with 'detail' set"))
    filters = utils.common_filters(marker=marker, limit=limit, sort_key=sort_key, sort_dir=sort_dir, fields=fields, detail=detail)
    path = '%s/volume/targets' % node_id
    if filters:
        path += '?' + '&'.join(filters)
    header_values = {'os_ironic_api_version': os_ironic_api_version, 'global_request_id': global_request_id}
    if limit is None:
        return self._list(self._path(path), response_key='targets', obj_class=volume_target.VolumeTarget, **header_values)
    else:
        return self._list_pagination(self._path(path), response_key='targets', limit=limit, obj_class=volume_target.VolumeTarget, **header_values)