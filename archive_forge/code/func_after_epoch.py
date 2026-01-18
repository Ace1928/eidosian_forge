import logging
import os
import tempfile
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from fastai.callback.core import Callback
import mlflow.tracking
from mlflow.fastai import log_model
from mlflow.utils.autologging_utils import ExceptionSafeClass, get_autologging_config
def after_epoch(self):
    """Log loss and other metrics values after each epoch"""

    def _is_float(x):
        try:
            float(x)
            return True
        except (ValueError, TypeError):
            return False
    if hasattr(self, 'lr_finder') or hasattr(self, 'gather_preds'):
        return
    metrics = self.recorder.log
    metrics = {k: v for k, v in zip(self.recorder.metric_names, metrics) if _is_float(v)}
    self.metrics_logger.record_metrics(metrics, step=metrics['epoch'])