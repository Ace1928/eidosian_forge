import logging
import math
import time
from .model import save_checkpoint
def module_checkpoint(mod, prefix, period=1, save_optimizer_states=False):
    """Callback to checkpoint Module to prefix every epoch.

    Parameters
    ----------
    mod : subclass of BaseModule
        The module to checkpoint.
    prefix : str
        The file prefix for this checkpoint.
    period : int
        How many epochs to wait before checkpointing. Defaults to 1.
    save_optimizer_states : bool
        Indicates whether or not to save optimizer states for continued training.

    Returns
    -------
    callback : function
        The callback function that can be passed as iter_end_callback to fit.
    """
    period = int(max(1, period))

    def _callback(iter_no, sym=None, arg=None, aux=None):
        """The checkpoint function."""
        if (iter_no + 1) % period == 0:
            mod.save_checkpoint(prefix, iter_no + 1, save_optimizer_states)
    return _callback