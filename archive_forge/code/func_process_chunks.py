import base64
import functools
import itertools
import logging
import os
import queue
import random
import sys
import threading
import time
from types import TracebackType
from typing import (
import requests
import wandb
from wandb import util
from wandb.sdk.internal import internal_api
from ..lib import file_stream_utils
def process_chunks(self, chunks: List[Chunk]) -> 'ProcessedBinaryChunk':
    data = b''.join([c.data for c in chunks])
    enc = base64.b64encode(data).decode('ascii')
    self._offset += len(data)
    return {'offset': self._offset, 'content': enc, 'encoding': 'base64'}