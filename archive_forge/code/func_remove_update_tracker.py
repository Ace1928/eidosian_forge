import abc
import collections
import dataclasses
import math
import typing
from typing import (
import weakref
import immutabledict
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt.python import hash_model_storage
from ortools.math_opt.python import model_storage
def remove_update_tracker(self, tracker: UpdateTracker):
    """Stops tracker from getting updates on changes to this model.

        An error will be raised if tracker was not created by this Model or if
        tracker has been previously removed.

        Using (via checkpoint or update) an UpdateTracker after it has been removed
        will result in an error.

        Args:
          tracker: The UpdateTracker to unregister.

        Raises:
          KeyError: The tracker was created by another model or was already removed.
        """
    self.storage.remove_update_tracker(tracker.storage_update_tracker)