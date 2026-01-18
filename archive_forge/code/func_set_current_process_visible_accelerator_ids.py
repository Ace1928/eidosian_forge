import os
import re
import glob
import requests
import logging
from typing import Dict, Optional, List, Tuple
from ray._private.accelerators.accelerator import AcceleratorManager
@staticmethod
def set_current_process_visible_accelerator_ids(visible_tpu_chips: List[str]) -> None:
    """Set TPU environment variables based on the provided visible_tpu_chips.

        To access a subset of the TPU visible chips, we must use a combination of
        environment variables that tells the compiler (via ML framework) the:
        - Visible chips
        - The physical bounds of chips per host
        - The host bounds within the context of a TPU pod.

        See: https://github.com/google/jax/issues/14977 for an example/more details.

        Args:
            visible_tpu_chips (List[str]): List of int representing TPU chips.
        """
    if os.environ.get(NOSET_TPU_VISIBLE_CHIPS_ENV_VAR):
        return
    num_visible_tpu_chips = len(visible_tpu_chips)
    if num_visible_tpu_chips == TPU_NUM_CHIPS_PER_HOST:
        return
    os.environ[TPUAcceleratorManager.get_visible_accelerator_ids_env_var()] = ','.join([str(i) for i in visible_tpu_chips])
    if num_visible_tpu_chips == 1:
        os.environ[TPU_CHIPS_PER_HOST_BOUNDS_ENV_VAR] = TPU_CHIPS_PER_HOST_BOUNDS_1_CHIP_CONFIG
        os.environ[TPU_HOST_BOUNDS_ENV_VAR] = TPU_SINGLE_HOST_BOUNDS
    elif num_visible_tpu_chips == 2:
        os.environ[TPU_CHIPS_PER_HOST_BOUNDS_ENV_VAR] = TPU_CHIPS_PER_HOST_BOUNDS_2_CHIP_CONFIG
        os.environ[TPU_HOST_BOUNDS_ENV_VAR] = TPU_SINGLE_HOST_BOUNDS
    elif num_visible_tpu_chips == 4:
        os.environ[TPU_CHIPS_PER_HOST_BOUNDS_ENV_VAR] = None
        os.environ[TPU_HOST_BOUNDS_ENV_VAR] = None