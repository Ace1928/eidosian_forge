from __future__ import annotations
import json
import logging
import os
from multiprocessing import Manager, Pool
from typing import TYPE_CHECKING
from monty.io import zopen
from monty.json import MontyDecoder, MontyEncoder
def serial_assimilate(self, root: str | Path) -> None:
    """Assimilate the entire subdirectory structure in rootpath serially."""
    valid_paths = []
    for parent, subdirs, files in os.walk(root):
        valid_paths.extend(self._drone.get_valid_paths((parent, subdirs, files)))
    data: list[str] = []
    total = len(valid_paths)
    for idx, path in enumerate(valid_paths, 1):
        new_data = self._drone.assimilate(path)
        self._data.append(new_data)
        logger.info(f'{idx}/{total} ({idx / total:.1%}) done')
    for json_str in data:
        self._data.append(json.loads(json_str, cls=MontyDecoder))