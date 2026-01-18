import os
import tempfile
from typing import TYPE_CHECKING, Optional, Union
from sklearn.base import BaseEstimator
import ray.cloudpickle as cpickle
from ray.train._internal.framework_checkpoint import FrameworkCheckpoint
from ray.util.annotations import PublicAPI
def get_estimator(self) -> BaseEstimator:
    """Retrieve the ``Estimator`` stored in this checkpoint."""
    with self.as_directory() as checkpoint_path:
        estimator_path = os.path.join(checkpoint_path, self.MODEL_FILENAME)
        with open(estimator_path, 'rb') as f:
            return cpickle.load(f)