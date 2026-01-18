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
def before_fit(self):
    from fastai.callback.all import ParamScheduler
    if hasattr(self, 'lr_finder') or hasattr(self, 'gather_preds'):
        return
    if self.is_fine_tune and len(self.opt.param_lists) == 1:
        _logger.warning('Using `fine_tune` with model which cannot be frozen. Current model have only one param group which makes it impossible to freeze. Because of this it will record some fitting params twice (overriding exception)')
    frozen = self.opt.frozen_idx != 0
    if frozen and self.is_fine_tune:
        self.freeze_prefix = 'freeze_'
        mlflow.log_param('frozen_idx', self.opt.frozen_idx)
    else:
        self.freeze_prefix = ''
    if isinstance(self.opt_func, partial):
        mlflow.log_param(self.freeze_prefix + 'opt_func', self.opt_func.keywords['opt'].__name__)
    else:
        mlflow.log_param(self.freeze_prefix + 'opt_func', self.opt_func.__name__)
    params_not_to_log = []
    for cb in self.cbs:
        if isinstance(cb, ParamScheduler):
            params_not_to_log = list(cb.scheds.keys())
            for param, f in cb.scheds.items():
                values = []
                for step in np.linspace(0, 1, num=100, endpoint=False):
                    values.append(f(step))
                values = np.array(values)
                mlflow.log_param(self.freeze_prefix + param + '_min', np.min(values, 0))
                mlflow.log_param(self.freeze_prefix + param + '_max', np.max(values, 0))
                mlflow.log_param(self.freeze_prefix + param + '_init', values[0])
                mlflow.log_param(self.freeze_prefix + param + '_final', values[-1])
                fig = plt.figure()
                plt.plot(values)
                plt.ylabel(param)
                with tempfile.TemporaryDirectory() as tempdir:
                    scheds_file = os.path.join(tempdir, self.freeze_prefix + param + '.png')
                    plt.savefig(scheds_file)
                    plt.close(fig)
                    mlflow.log_artifact(local_path=scheds_file)
            break
    for param in self.opt.hypers[0]:
        if param not in params_not_to_log:
            mlflow.log_param(self.freeze_prefix + param, [h[param] for h in self.opt.hypers])
    if hasattr(self.opt, 'true_wd'):
        mlflow.log_param(self.freeze_prefix + 'true_wd', self.opt.true_wd)
    if hasattr(self.opt, 'bn_wd'):
        mlflow.log_param(self.freeze_prefix + 'bn_wd', self.opt.bn_wd)
    if hasattr(self.opt, 'train_bn'):
        mlflow.log_param(self.freeze_prefix + 'train_bn', self.opt.train_bn)