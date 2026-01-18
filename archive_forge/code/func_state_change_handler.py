import time
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import monitor as vrrp_monitor
from os_ken.services.protocols.vrrp import router as vrrp_router
@handler.set_ev_cls(vrrp_event.EventVRRPStateChanged)
def state_change_handler(self, ev):
    instance = self._instances.get(ev.instance_name, None)
    assert instance is not None
    instance.state_changed(ev.new_state)
    if ev.old_state and ev.new_state == vrrp_event.VRRP_STATE_INITIALIZE:
        self.shutdown.put(instance)