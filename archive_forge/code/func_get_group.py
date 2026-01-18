import json
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional
def get_group(self, filename: str) -> Optional[Dict[str, str]]:
    grp_filename = f'__grp__{filename}'
    if not self.has_file(grp_filename):
        return None
    grp_filepath = self._make_path(grp_filename)
    with open(grp_filepath) as f:
        grp_data = json.load(f)
    child_paths = grp_data.get('child_paths', None)
    if child_paths is None:
        return None
    result = {}
    for c in child_paths:
        p = self._make_path(c)
        if os.path.exists(p):
            result[c] = p
    return result