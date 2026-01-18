from openstack.common import tag
from openstack.network.v2 import _base
from openstack import resource
@classmethod
def find_available(cls, session):
    for ip in cls.list(session):
        if not ip.port_id:
            return ip
    return None