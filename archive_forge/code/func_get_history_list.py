import logging
import os
from oslo_utils import strutils
from ironicclient.common import base
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import volume_connector
from ironicclient.v1 import volume_target
def get_history_list(self, node_ident, detail=False, os_ironic_api_version=None, global_request_id=None):
    """Get node history event list.

        Provides the ability to query a node event history list from
        the API and return the API response to the caller.

        Requires API version 1.78.

        :param node_ident: The name or UUID of the node.
        :param detail: If detailed data should be returned in the
                       event list entry. Default False.
        :param os_ironic_api_version: String version (e.g. "1.35") to use for
            the request.  If not specified, the client's default is used.
        :param global_request_id: String containing global request ID header
            value (in form "req-<UUID>") to use for the request.
        """
    path = '%s/history' % node_ident
    if detail:
        path = path + '/detail'
    return self._list_primitives(self._path(path), 'history', os_ironic_api_version=os_ironic_api_version, global_request_id=global_request_id)