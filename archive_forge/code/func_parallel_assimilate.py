from __future__ import annotations
import json
import logging
import os
from multiprocessing import Manager, Pool
from typing import TYPE_CHECKING
from monty.io import zopen
from monty.json import MontyDecoder, MontyEncoder
def parallel_assimilate(self, rootpath):
    """Assimilate the entire subdirectory structure in rootpath."""
    logger.info('Scanning for valid paths...')
    valid_paths = []
    for parent, subdirs, files in os.walk(rootpath):
        valid_paths.extend(self._drone.get_valid_paths((parent, subdirs, files)))
    manager = Manager()
    data = manager.list()
    status = manager.dict()
    status['count'] = 0
    status['total'] = len(valid_paths)
    logger.info(f'{len(valid_paths)} valid paths found.')
    with Pool(self._num_drones) as pool:
        pool.map(order_assimilation, ((path, self._drone, data, status) for path in valid_paths))
        for string in data:
            self._data.append(json.loads(string, cls=MontyDecoder))