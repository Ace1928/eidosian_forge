import time
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import monitor as vrrp_monitor
from os_ken.services.protocols.vrrp import router as vrrp_router
@handler.set_ev_cls(vrrp_event.EventVRRPConfigRequest)
def config_request_handler(self, ev):
    config = ev.config
    interface = ev.interface
    name = self._instance_name(interface, config.vrid, config.is_ipv6)
    if name in self._instances:
        rep = vrrp_event.EventVRRPConfigReply(None, interface, config)
        self.reply_to_request(ev, rep)
        return
    statistics = VRRPStatistics(name, config.resource_id, config.statistics_interval)
    monitor = vrrp_monitor.VRRPInterfaceMonitor.factory(interface, config, name, statistics, *self._args, **self._kwargs)
    router = vrrp_router.VRRPRouter.factory(name, monitor.name, interface, config, statistics, *self._args, **self._kwargs)
    self.register_observer(vrrp_event.EventVRRPShutdownRequest, router.name)
    router.register_observer(vrrp_event.EventVRRPStateChanged, monitor.name)
    router.register_observer(vrrp_event.EventVRRPTransmitRequest, monitor.name)
    monitor.register_observer(vrrp_event.EventVRRPReceived, router.name)
    instance = VRRPInstance(name, monitor.name, config, interface)
    self._instances[name] = instance
    monitor.start()
    router.start()
    rep = vrrp_event.EventVRRPConfigReply(instance.name, interface, config)
    self.reply_to_request(ev, rep)