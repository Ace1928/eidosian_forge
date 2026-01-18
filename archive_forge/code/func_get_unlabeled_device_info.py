from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import math
import re
import os
def get_unlabeled_device_info(device, unit):
    """
    Fetches device information directly from the kernel and it is used when
    parted cannot work because of a missing label. It always returns a 'unknown'
    label.
    """
    device_name = os.path.basename(device)
    base = '/sys/block/%s' % device_name
    vendor = read_record(base + '/device/vendor', 'Unknown')
    model = read_record(base + '/device/model', 'model')
    logic_block = int(read_record(base + '/queue/logical_block_size', 0))
    phys_block = int(read_record(base + '/queue/physical_block_size', 0))
    size_bytes = int(read_record(base + '/size', 0)) * logic_block
    size, unit = format_disk_size(size_bytes, unit)
    return {'generic': {'dev': device, 'table': 'unknown', 'size': size, 'unit': unit, 'logical_block': logic_block, 'physical_block': phys_block, 'model': '%s %s' % (vendor, model)}, 'partitions': []}