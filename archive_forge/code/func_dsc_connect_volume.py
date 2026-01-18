import glob
import http.client
import os
import re
import tempfile
import time
import traceback
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator.connectors import base
from os_brick.privileged import lightos as priv_lightos
from os_brick import utils
def dsc_connect_volume(self, connection_info):
    if not self.dsc_need_connect(connection_info):
        return
    subsysnqn = connection_info['subsysnqn']
    uuid = connection_info['uuid']
    hostnqn = utils.get_host_nqn()
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as dscfile:
        dscfile.write('# os_brick connector dsc file for LightOS volume: {}\n'.format(uuid))
        for ip, node in connection_info['lightos_nodes'].items():
            transport = node['transport_type']
            host = node['target_portal']
            port = node['target_port']
            dscfile.write('-t {} -a {} -s {} -q {} -n {}\n'.format(transport, host, port, hostnqn, subsysnqn))
        dscfile.flush()
        try:
            dest_name = self.dsc_file_name(uuid)
            priv_lightos.move_dsc_file(dscfile.name, dest_name)
        except Exception:
            LOG.warning('LIGHTOS: Failed to create dsc file for connection with uuid:%s', uuid)
            raise