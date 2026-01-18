from __future__ import annotations
import asyncio
import copy
import os
import random
import time
import traceback
import uuid
from collections import defaultdict
from queue import Queue as ThreadQueue
from typing import TYPE_CHECKING
import fastapi
from typing_extensions import Literal
from gradio import route_utils, routes
from gradio.data_classes import (
from gradio.exceptions import Error
from gradio.helpers import TrackedIterable
from gradio.server_messages import (
from gradio.utils import LRUCache, run_coro_in_background, safe_get_lock, set_task_name
def set_event_queue_per_concurrency_id(self):
    for block_fn in self.block_fns:
        concurrency_id = block_fn.concurrency_id
        concurrency_limit: int | None
        if block_fn.concurrency_limit == 'default':
            concurrency_limit = self.default_concurrency_limit
        else:
            concurrency_limit = block_fn.concurrency_limit
        if concurrency_id not in self.event_queue_per_concurrency_id:
            self.event_queue_per_concurrency_id[concurrency_id] = EventQueue(concurrency_id, concurrency_limit)
        elif concurrency_limit is not None:
            existing_event_queue = self.event_queue_per_concurrency_id[concurrency_id]
            if existing_event_queue.concurrency_limit is None or concurrency_limit < existing_event_queue.concurrency_limit:
                existing_event_queue.concurrency_limit = concurrency_limit