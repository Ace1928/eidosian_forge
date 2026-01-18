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
def after_fit(self):
    from fastai.callback.all import SaveModelCallback
    if hasattr(self, 'lr_finder') or hasattr(self, 'gather_preds'):
        return
    for cb in self.cbs:
        if isinstance(cb, SaveModelCallback):
            cb('after_fit')
    if self.log_models:
        registered_model_name = get_autologging_config(mlflow.fastai.FLAVOR_NAME, 'registered_model_name', None)
        log_model(self.learn, artifact_path='model', registered_model_name=registered_model_name)