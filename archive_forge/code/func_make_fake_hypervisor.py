import datetime
import hashlib
import json
import uuid
from openstack.cloud import meta
from openstack.orchestration.util import template_format
from openstack import utils
def make_fake_hypervisor(id, name):
    return json.loads(json.dumps({'id': id, 'hypervisor_hostname': name, 'state': 'up', 'status': 'enabled', 'cpu_info': {'arch': 'x86_64', 'model': 'Nehalem', 'vendor': 'Intel', 'features': ['pge', 'clflush'], 'topology': {'cores': 1, 'threads': 1, 'sockets': 4}}, 'current_workload': 0, 'status': 'enabled', 'state': 'up', 'disk_available_least': 0, 'host_ip': '1.1.1.1', 'free_disk_gb': 1028, 'free_ram_mb': 7680, 'hypervisor_type': 'fake', 'hypervisor_version': 1000, 'local_gb': 1028, 'local_gb_used': 0, 'memory_mb': 8192, 'memory_mb_used': 512, 'running_vms': 0, 'service': {'host': 'host1', 'id': 7, 'disabled_reason': None}, 'vcpus': 1, 'vcpus_used': 0}))