from os_ken.services.protocols.bgp.signals import SignalBus
def adj_down(self, peer):
    return self.emit_signal(self.BGP_ADJ_DOWN, {'peer': peer})