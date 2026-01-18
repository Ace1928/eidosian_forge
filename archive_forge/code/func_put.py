import json
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional
def put(self, data, filename, binary=True) -> str:
    if not self.cache_dir:
        raise RuntimeError('Could not create or locate cache dir')
    binary = isinstance(data, bytes)
    if not binary:
        data = str(data)
    assert self.lock_path is not None
    filepath = self._make_path(filename)
    rnd_id = random.randint(0, 1000000)
    pid = os.getpid()
    temp_path = f'{filepath}.tmp.pid_{pid}_{rnd_id}'
    mode = 'wb' if binary else 'w'
    with open(temp_path, mode) as f:
        f.write(data)
    os.replace(temp_path, filepath)
    return filepath