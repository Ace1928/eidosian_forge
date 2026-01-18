import collections
import errno
import uuid
from ovs import jsonrpc
from ovs import poller
from ovs import reconnect
from ovs import stream
from ovs import timeval
from ovs.db import idl
from os_ken.base import app_manager
from os_ken.lib import hub
from os_ken.services.protocols.ovsdb import event
from os_ken.services.protocols.ovsdb import model
def dictify(row):
    if row is None:
        return
    result = {}
    for key, value in row._data.items():
        result[key] = value.to_python(_uuid_to_row)
        hub.sleep(0)
    return result