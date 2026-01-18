from __future__ import absolute_import
import datetime as dt
import json
import math
import time
import typing
from typing import Optional, Callable
from google.cloud.pubsub_v1.subscriber._protocol import requests
from google.cloud.pubsub_v1.subscriber import futures
from google.cloud.pubsub_v1.subscriber.exceptions import AcknowledgeStatus
@property
def publish_time(self) -> 'datetime.datetime':
    """Return the time that the message was originally published.

        Returns:
            datetime.datetime: The date and time that the message was
            published.
        """
    return self._publish_time