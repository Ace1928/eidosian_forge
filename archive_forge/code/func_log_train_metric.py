import logging
import math
import time
from .model import save_checkpoint
def log_train_metric(period, auto_reset=False):
    """Callback to log the training evaluation result every period.

    Parameters
    ----------
    period : int
        The number of batch to log the training evaluation metric.
    auto_reset : bool
        Reset the metric after each log.

    Returns
    -------
    callback : function
        The callback function that can be passed as iter_epoch_callback to fit.
    """

    def _callback(param):
        """The checkpoint function."""
        if param.nbatch % period == 0 and param.eval_metric is not None:
            name_value = param.eval_metric.get_name_value()
            for name, value in name_value:
                logging.info('Iter[%d] Batch[%d] Train-%s=%f', param.epoch, param.nbatch, name, value)
            if auto_reset:
                param.eval_metric.reset_local()
    return _callback