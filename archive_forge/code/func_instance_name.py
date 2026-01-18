from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.lib.packet import packet
from os_ken.lib.packet import vlan
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import event as vrrp_event
@classmethod
def instance_name(cls, interface, vrid):
    return '%s-%s-%d' % (cls.__name__, str(interface), vrid)