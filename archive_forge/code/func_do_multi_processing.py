import dataclasses
import json
import warnings
from dataclasses import dataclass, field
from time import time
from typing import List
from ..utils import logging
@property
def do_multi_processing(self):
    if not self.multi_process:
        return False
    elif self.is_tpu:
        logger.info('Multiprocessing is currently not possible on TPU.')
        return False
    else:
        return True