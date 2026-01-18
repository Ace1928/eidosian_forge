import os
import time
import warnings
import numpy as np
from ....metric import CompositeEvalMetric, EvalMetric
from ....metric import Loss as metric_loss
from .utils import _check_metrics
def epoch_begin(self, estimator, *args, **kwargs):
    if isinstance(self.log_interval, int) or self.log_interval == 'epoch':
        is_training = False
        for metric in self.metrics:
            if 'training' in metric.name:
                is_training = True
        self.epoch_start = time.time()
        if is_training:
            estimator.logger.info('[Epoch %d] Begin, current learning rate: %.4f', self.current_epoch, estimator.trainer.learning_rate)
        else:
            estimator.logger.info('Validation Begin')