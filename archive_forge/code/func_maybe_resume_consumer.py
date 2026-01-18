from __future__ import division
import collections
import functools
import itertools
import logging
import threading
import typing
from typing import Any, Dict, Callable, Iterable, List, Optional, Set, Tuple
import uuid
import grpc  # type: ignore
from google.api_core import bidi
from google.api_core import exceptions
from google.cloud.pubsub_v1 import types
from google.cloud.pubsub_v1.subscriber._protocol import dispatcher
from google.cloud.pubsub_v1.subscriber._protocol import heartbeater
from google.cloud.pubsub_v1.subscriber._protocol import histogram
from google.cloud.pubsub_v1.subscriber._protocol import leaser
from google.cloud.pubsub_v1.subscriber._protocol import messages_on_hold
from google.cloud.pubsub_v1.subscriber._protocol import requests
from google.cloud.pubsub_v1.subscriber.exceptions import (
import google.cloud.pubsub_v1.subscriber.message
from google.cloud.pubsub_v1.subscriber import futures
from google.cloud.pubsub_v1.subscriber.scheduler import ThreadScheduler
from google.pubsub_v1 import types as gapic_types
from google.rpc.error_details_pb2 import ErrorInfo  # type: ignore
from google.rpc import code_pb2  # type: ignore
from google.rpc import status_pb2
def maybe_resume_consumer(self) -> None:
    """Check the load and held messages and resume the consumer if needed.

        If there are messages held internally, release those messages before
        resuming the consumer. That will avoid leaser overload.
        """
    with self._pause_resume_lock:
        if self._consumer is None or not self._consumer.is_paused:
            return
        _LOGGER.debug('Current load: %.2f', self.load)
        self._maybe_release_messages()
        if self.load < _RESUME_THRESHOLD:
            _LOGGER.debug('Current load is %.2f, resuming consumer.', self.load)
            self._consumer.resume()
        else:
            _LOGGER.debug('Did not resume, current load is %.2f.', self.load)