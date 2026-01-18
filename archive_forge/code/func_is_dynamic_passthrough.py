import copy
import logging
import os
from ray.autoscaler._private.event_system import CreateClusterEvent, global_event_system
from ray.autoscaler._private.util import check_legacy_fields
def is_dynamic_passthrough(node_config):
    if 'gpu_config' in node_config:
        gpu_config = node_config['gpu_config']
        if gpu_config and gpu_config['dynamic_pci_passthrough']:
            return True
    return False