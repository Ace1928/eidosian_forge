import abc
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import api as vrrp_api
def preempt_delay(self, ev):
    self.vrrp_router.logger.warning('%s preempt_delay', self.__class__.__name__)
    self._master_down()