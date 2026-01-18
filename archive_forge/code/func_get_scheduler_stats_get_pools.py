from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def get_scheduler_stats_get_pools(self, **kw):
    stats = [{'name': 'ubuntu@lvm#backend_name', 'capabilities': {'pool_name': 'backend_name', 'QoS_support': False, 'timestamp': '2014-11-21T18:15:28.141161', 'allocated_capacity_gb': 0, 'volume_backend_name': 'backend_name', 'free_capacity_gb': 7.01, 'driver_version': '2.0.0', 'total_capacity_gb': 10.01, 'reserved_percentage': 0, 'vendor_name': 'Open Source', 'storage_protocol': 'iSCSI'}}]
    return (200, {}, {'pools': stats})