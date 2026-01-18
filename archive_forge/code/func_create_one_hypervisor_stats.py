from unittest import mock
from openstackclient.compute.v2 import hypervisor_stats
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
def create_one_hypervisor_stats(attrs=None):
    """Create a fake hypervisor stats.

    :param dict attrs:
        A dictionary with all attributes
    :return:
        A dictionary that contains hypervisor stats information keys
    """
    attrs = attrs or {}
    stats_info = {'count': 2, 'current_workload': 0, 'disk_available_least': 50, 'free_disk_gb': 100, 'free_ram_mb': 23000, 'local_gb': 100, 'local_gb_used': 0, 'memory_mb': 23800, 'memory_mb_used': 1400, 'running_vms': 3, 'vcpus': 8, 'vcpus_used': 3}
    stats_info.update(attrs)
    return stats_info