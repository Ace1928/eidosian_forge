import contextlib
import functools
import fasteners
from oslo_utils import reflection
from oslo_utils import uuidutils
import tenacity
from taskflow import exceptions
from taskflow import logging
from taskflow.persistence.backends import impl_memory
from taskflow.persistence import models
from taskflow import retry
from taskflow import states
from taskflow import task
from taskflow.utils import misc
@fasteners.read_locked
def get_retry_histories(self):
    """Fetch all retrys histories."""
    histories = []
    for ad in self._flowdetail:
        if isinstance(ad, models.RetryDetail):
            histories.append((ad.name, self._translate_into_history(ad)))
    return histories