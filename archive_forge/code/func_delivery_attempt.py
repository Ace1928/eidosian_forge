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
def delivery_attempt(self) -> Optional[int]:
    """The delivery attempt counter is 1 + (the sum of number of NACKs
        and number of ack_deadline exceeds) for this message. It is set to None
        if a DeadLetterPolicy is not set on the subscription.

        A NACK is any call to ModifyAckDeadline with a 0 deadline. An ack_deadline
        exceeds event is whenever a message is not acknowledged within
        ack_deadline. Note that ack_deadline is initially
        Subscription.ackDeadlineSeconds, but may get extended automatically by
        the client library.

        The first delivery of a given message will have this value as 1. The value
        is calculated at best effort and is approximate.

        Returns:
            Optional[int]: The delivery attempt counter or ``None``.
        """
    return self._delivery_attempt