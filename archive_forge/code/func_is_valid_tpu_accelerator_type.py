import os
import re
import glob
import requests
import logging
from typing import Dict, Optional, List, Tuple
from ray._private.accelerators.accelerator import AcceleratorManager
@staticmethod
def is_valid_tpu_accelerator_type(tpu_accelerator_type: str) -> bool:
    """Check whether the tpu accelerator_type is formatted correctly.

        The accelerator_type field follows a form of v{generation}-{cores/chips}.

        See the following for more information:
        https://cloud.google.com/sdk/gcloud/reference/compute/tpus/tpu-vm/accelerator-types/describe

        Args:
            tpu_accelerator_type: The string representation of the accelerator type
                to be checked for validity.

        Returns:
            True if it's valid, false otherwise.
        """
    expected_pattern = re.compile('^v\\d+[a-zA-Z]*-\\d+$')
    if not expected_pattern.match(tpu_accelerator_type):
        return False
    return True