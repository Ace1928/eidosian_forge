import abc
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import api as vrrp_api
@handler.set_ev_handler(_EventStatisticsOut)
def statistics_handler(self, ev):
    self.stats_out_timer.start(self.statistics.statistics_interval)