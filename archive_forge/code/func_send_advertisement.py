import abc
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import api as vrrp_api
def send_advertisement(self, release=False):
    if self.vrrp is None:
        config = self.config
        max_adver_int = vrrp.vrrp.sec_to_max_adver_int(config.version, config.advertisement_interval)
        self.vrrp = vrrp.vrrp.create_version(config.version, vrrp.VRRP_TYPE_ADVERTISEMENT, config.vrid, config.priority, max_adver_int, config.ip_addresses)
    vrrp_ = self.vrrp
    if release:
        vrrp_ = vrrp_.create(vrrp_.type, vrrp_.vrid, vrrp.VRRP_PRIORITY_RELEASE_RESPONSIBILITY, vrrp_.max_adver_int, vrrp_.ip_addresses)
    if self.vrrp.priority == 0:
        self.statistics.tx_vrrp_zero_prio_packets += 1
    interface = self.interface
    packet_ = vrrp_.create_packet(interface.primary_ip_address, interface.vlan_id)
    packet_.serialize()
    vrrp_api.vrrp_transmit(self, self.monitor_name, packet_.data)
    self.statistics.tx_vrrp_packets += 1