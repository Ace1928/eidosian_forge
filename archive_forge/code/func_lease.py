from __future__ import absolute_import
from __future__ import division
import functools
import itertools
import logging
import math
import time
import threading
import typing
from typing import List, Optional, Sequence, Union
import warnings
from google.api_core.retry import exponential_sleep_generator
from google.cloud.pubsub_v1.subscriber._protocol import helper_threads
from google.cloud.pubsub_v1.subscriber._protocol import requests
from google.cloud.pubsub_v1.subscriber.exceptions import (
def lease(self, items: Sequence[requests.LeaseRequest]) -> None:
    """Add the given messages to lease management.

        Args:
            items: The items to lease.
        """
    assert self._manager.leaser is not None
    self._manager.leaser.add(items)
    self._manager.maybe_pause_consumer()