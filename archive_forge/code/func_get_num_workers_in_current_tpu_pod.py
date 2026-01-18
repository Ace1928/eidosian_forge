import os
import re
import glob
import requests
import logging
from typing import Dict, Optional, List, Tuple
from ray._private.accelerators.accelerator import AcceleratorManager
@staticmethod
def get_num_workers_in_current_tpu_pod() -> Optional[int]:
    """Return the total number of workers in a TPU pod."""
    tpu_pod_type = TPUAcceleratorManager._get_current_node_tpu_pod_type()
    if tpu_pod_type:
        version = tpu_pod_type.split('-')[0]
        num_chips_or_cores = int(tpu_pod_type.split('-')[1])
        if version in TPU_VERSIONS_WITH_MULTIPLE_CORES_PER_CHIP:
            return num_chips_or_cores // 8
        else:
            return num_chips_or_cores // 4
    else:
        logging.debug('Could not get num workers in TPU pod.')
        return None