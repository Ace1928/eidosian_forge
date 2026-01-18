from concurrent import futures
import datetime
import json
import logging
import os
import time
import urllib
from absl import flags
def state(self):
    """Return state of the TPU."""
    return self._get_tpu_property('state')