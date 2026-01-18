from os_ken.services.protocols.bgp.signals import SignalBus
def bgp_notification_received(self, peer, notification):
    return self.emit_signal(self.BGP_NOTIFICATION_RECEIVED + (peer,), notification)