import os
from os_win import utilsfactory
from os_brick.initiator.windows import base as win_conn_base
from os_brick.remotefs import windows_remotefs as remotefs
from os_brick import utils
def ensure_share_mounted(self, connection_properties):
    export_path = self._get_export_path(connection_properties)
    mount_options = connection_properties.get('options')
    self._remotefsclient.mount(export_path, mount_options)