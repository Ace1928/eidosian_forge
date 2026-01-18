import time
from typing import List, Optional, Tuple
import os
import ray
import numpy as np
from contextlib import contextmanager
@contextmanager
def ray_setup_and_teardown(**init_args):
    ray.init(**init_args)
    try:
        yield None
    finally:
        ray.shutdown()