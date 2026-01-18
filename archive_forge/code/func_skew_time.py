import abc
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import api as vrrp_api
@property
def skew_time(self):
    config = self.config
    version = config.version
    priority = config.priority
    if config.version == vrrp.VRRP_VERSION_V2:
        return (256.0 - priority) / 256.0
    if config.version == vrrp.VRRP_VERSION_V3:
        return (256.0 - priority) * self.master_adver_interval / 256.0
    raise ValueError('unknown vrrp version %d' % version)