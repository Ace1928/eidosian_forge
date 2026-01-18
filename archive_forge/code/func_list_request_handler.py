import time
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import monitor as vrrp_monitor
from os_ken.services.protocols.vrrp import router as vrrp_router
@handler.set_ev_cls(vrrp_event.EventVRRPListRequest)
def list_request_handler(self, ev):
    instance_name = ev.instance_name
    if instance_name is None:
        instance_list = [vrrp_event.VRRPInstance(instance.name, instance.monitor_name, instance.config, instance.interface, instance.state) for instance in self._instances.values()]
    else:
        instance = self._instances.get(instance_name, None)
        if instance is None:
            instance_list = []
        else:
            instance_list = [vrrp_event.VRRPInstance(instance_name, instance.monitor_name, instance.config, instance.interface, instance.state)]
    vrrp_list = vrrp_event.EventVRRPListReply(instance_list)
    self.reply_to_request(ev, vrrp_list)