import logging
import math
import threading
from botocore.retries import bucket, standard, throttling
@property
def measured_rate(self):
    return self._measured_rate