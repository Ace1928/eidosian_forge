from __future__ import annotations
import json
import logging
import os
from multiprocessing import Manager, Pool
from typing import TYPE_CHECKING
from monty.io import zopen
from monty.json import MontyDecoder, MontyEncoder
def order_assimilation(args):
    """Internal helper method for BorgQueen to process assimilation."""
    path, drone, data, status = args
    new_data = drone.assimilate(path)
    if new_data:
        data.append(json.dumps(new_data, cls=MontyEncoder))
    status['count'] += 1
    count = status['count']
    total = status['total']
    logger.info(f'{count}/{total} ({count / total:.2%}) done')