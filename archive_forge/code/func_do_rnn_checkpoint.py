import warnings
from ..model import save_checkpoint, load_checkpoint
from .rnn_cell import BaseRNNCell
def do_rnn_checkpoint(cells, prefix, period=1):
    """Make a callback to checkpoint Module to prefix every epoch.
    unpacks weights used by cells before saving.

    Parameters
    ----------
    cells : mxnet.rnn.RNNCell or list of RNNCells
        The RNN cells used by this symbol.
    prefix : str
        The file prefix to checkpoint to
    period : int
        How many epochs to wait before checkpointing. Default is 1.

    Returns
    -------
    callback : function
        The callback function that can be passed as iter_end_callback to fit.
    """
    period = int(max(1, period))

    def _callback(iter_no, sym=None, arg=None, aux=None):
        """The checkpoint function."""
        if (iter_no + 1) % period == 0:
            save_rnn_checkpoint(cells, prefix, iter_no + 1, sym, arg, aux)
    return _callback