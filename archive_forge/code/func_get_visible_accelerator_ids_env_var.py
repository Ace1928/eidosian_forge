import os
import re
import glob
import requests
import logging
from typing import Dict, Optional, List, Tuple
from ray._private.accelerators.accelerator import AcceleratorManager
@staticmethod
def get_visible_accelerator_ids_env_var() -> str:
    return TPU_VISIBLE_CHIPS_ENV_VAR