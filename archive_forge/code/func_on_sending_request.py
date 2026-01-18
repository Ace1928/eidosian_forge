import logging
import math
import threading
from botocore.retries import bucket, standard, throttling
def on_sending_request(self, request, **kwargs):
    if self._enabled:
        self._token_bucket.acquire()