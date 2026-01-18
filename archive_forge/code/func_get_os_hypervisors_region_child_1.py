import copy
import datetime
import re
from unittest import mock
from urllib import parse
from oslo_utils import strutils
import novaclient
from novaclient import api_versions
from novaclient import client as base_client
from novaclient import exceptions
from novaclient.tests.unit import fakes
from novaclient.tests.unit import utils
from novaclient.v2 import client
def get_os_hypervisors_region_child_1(self, **kw):
    return (200, {}, {'hypervisor': {'id': 'region!child@1', 'service': {'id': 1, 'host': 'compute1'}, 'vcpus': 4, 'memory_mb': 10 * 1024, 'local_gb': 250, 'vcpus_used': 2, 'memory_mb_used': 5 * 1024, 'local_gb_used': 125, 'hypervisor_type': 'xen', 'hypervisor_version': 3, 'hypervisor_hostname': 'hyper1', 'free_ram_mb': 5 * 1024, 'free_disk_gb': 125, 'current_workload': 2, 'running_vms': 2, 'cpu_info': 'cpu_info', 'disk_available_least': 100}})