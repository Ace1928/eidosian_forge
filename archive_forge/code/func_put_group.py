import json
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional
def put_group(self, filename: str, group: Dict[str, str]) -> str:
    if not self.cache_dir:
        raise RuntimeError('Could not create or locate cache dir')
    grp_contents = json.dumps({'child_paths': sorted(list(group.keys()))})
    grp_filename = f'__grp__{filename}'
    return self.put(grp_contents, grp_filename, binary=False)