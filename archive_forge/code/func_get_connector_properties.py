import collections
import time
from os_win import exceptions as os_win_exc
from os_win import utilsfactory
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator.windows import base as win_conn_base
from os_brick import utils
@staticmethod
def get_connector_properties(*args, **kwargs):
    props = {}
    fc_utils = utilsfactory.get_fc_utils()
    fc_utils.refresh_hba_configuration()
    fc_hba_ports = fc_utils.get_fc_hba_ports()
    if fc_hba_ports:
        wwnns = []
        wwpns = []
        for port in fc_hba_ports:
            wwnns.append(port['node_name'])
            wwpns.append(port['port_name'])
        props['wwpns'] = wwpns
        props['wwnns'] = list(set(wwnns))
    return props