import datetime
import random
import time
from base64 import b64decode, b64encode
from typing import Optional
import etcd  # type: ignore[import]
from torch.distributed import Store
def cas_delay():
    time.sleep(random.uniform(0, 0.1))