import os
import re
import glob
import requests
import logging
from typing import Dict, Optional, List, Tuple
from ray._private.accelerators.accelerator import AcceleratorManager
@staticmethod
def get_resource_name() -> str:
    return 'TPU'