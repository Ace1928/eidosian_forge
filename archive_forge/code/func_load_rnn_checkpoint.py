import warnings
from ..model import save_checkpoint, load_checkpoint
from .rnn_cell import BaseRNNCell
def load_rnn_checkpoint(cells, prefix, epoch):
    """Load model checkpoint from file.
    Pack weights after loading.

    Parameters
    ----------
    cells : mxnet.rnn.RNNCell or list of RNNCells
        The RNN cells used by this symbol.
    prefix : str
        Prefix of model name.
    epoch : int
        Epoch number of model we would like to load.

    Returns
    -------
    symbol : Symbol
        The symbol configuration of computation network.
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.

    Notes
    -----
    - symbol will be loaded from ``prefix-symbol.json``.
    - parameters will be loaded from ``prefix-epoch.params``.
    """
    sym, arg, aux = load_checkpoint(prefix, epoch)
    if isinstance(cells, BaseRNNCell):
        cells = [cells]
    for cell in cells:
        arg = cell.pack_weights(arg)
    return (sym, arg, aux)