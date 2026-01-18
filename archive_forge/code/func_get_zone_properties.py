from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
def get_zone_properties(self, zone_name):
    zone = self._get_zone(zone_name, ignore_missing=False)
    zone_properties = {}
    zone_properties['zone_type'] = zone.ZoneType
    zone_properties['ds_integrated'] = zone.DsIntegrated
    zone_properties['data_file_name'] = zone.DataFile
    zone_properties['master_servers'] = zone.MasterServers or []
    return zone_properties