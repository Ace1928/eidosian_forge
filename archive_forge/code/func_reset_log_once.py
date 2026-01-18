from collections import defaultdict, namedtuple
import gc
import os
import re
import time
import tracemalloc
from typing import Callable, List, Optional
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
def reset_log_once(key):
    """Resets log_once for the provided key."""
    _logged.discard(key)