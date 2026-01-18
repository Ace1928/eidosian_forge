import logging
import math
import time
from .model import save_checkpoint
def do_checkpoint(prefix, period=1):
    """A callback that saves a model checkpoint every few epochs.
    Each checkpoint is made up of a couple of binary files: a model description file and a
    parameters (weights and biases) file. The model description file is named
    `prefix`--symbol.json and the parameters file is named `prefix`-`epoch_number`.params

    Parameters
    ----------
    prefix : str
        Prefix for the checkpoint filenames.
    period : int, optional
        Interval (number of epochs) between checkpoints. Default `period` is 1.

    Returns
    -------
    callback : function
        A callback function that can be passed as `epoch_end_callback` to fit.

    Example
    -------
    >>> module.fit(iterator, num_epoch=n_epoch,
    ... epoch_end_callback  = mx.callback.do_checkpoint("mymodel", 1))
    Start training with [cpu(0)]
    Epoch[0] Resetting Data Iterator
    Epoch[0] Time cost=0.100
    Saved checkpoint to "mymodel-0001.params"
    Epoch[1] Resetting Data Iterator
    Epoch[1] Time cost=0.060
    Saved checkpoint to "mymodel-0002.params"
    """
    period = int(max(1, period))

    def _callback(iter_no, sym, arg, aux):
        """The checkpoint function."""
        if (iter_no + 1) % period == 0:
            save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
    return _callback