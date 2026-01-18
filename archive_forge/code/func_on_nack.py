from abc import ABC, abstractmethod
from typing import Callable
from google.api_core.exceptions import FailedPrecondition
from google.pubsub_v1 import PubsubMessage
def on_nack(self, message: PubsubMessage, ack: Callable[[], None]):
    raise FailedPrecondition('You may not nack messages by default when using a PubSub Lite client. See NackHandler for how to customize this.')