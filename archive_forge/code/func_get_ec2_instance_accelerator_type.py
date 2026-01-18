import os
import sys
import json
import logging
import subprocess
from typing import Optional, List, Tuple
from ray._private.accelerators.accelerator import AcceleratorManager
@staticmethod
def get_ec2_instance_accelerator_type(instance_type: str, instances: dict) -> Optional[str]:
    from ray.util.accelerators import AWS_NEURON_CORE
    return AWS_NEURON_CORE