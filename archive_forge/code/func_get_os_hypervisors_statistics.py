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
def get_os_hypervisors_statistics(self, **kw):
    return (200, {}, {'hypervisor_statistics': {'count': 2, 'vcpus': 8, 'memory_mb': 20 * 1024, 'local_gb': 500, 'vcpus_used': 4, 'memory_mb_used': 10 * 1024, 'local_gb_used': 250, 'free_ram_mb': 10 * 1024, 'free_disk_gb': 250, 'current_workload': 4, 'running_vms': 4, 'disk_available_least': 200}})