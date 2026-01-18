import os
import sys
import json
import logging
import subprocess
from typing import Optional, List, Tuple
from ray._private.accelerators.accelerator import AcceleratorManager
@staticmethod
def get_ec2_instance_num_accelerators(instance_type: str, instances: dict) -> Optional[int]:
    return AWS_NEURON_INSTANCE_MAP.get(instance_type.lower(), None)