from ._internal import NDArrayBase
from ..base import _Null
def quantized_rnn(data=None, parameters=None, state=None, state_cell=None, data_scale=None, data_shift=None, state_size=_Null, num_layers=_Null, bidirectional=_Null, mode=_Null, p=_Null, state_outputs=_Null, projection_size=_Null, lstm_state_clip_min=_Null, lstm_state_clip_max=_Null, lstm_state_clip_nan=_Null, use_sequence_length=_Null, out=None, name=None, **kwargs):
    """RNN operator for input data type of uint8. The weight of each gates is converted
    to int8, while bias is accumulated in type float32. The hidden state and cell state are in type
    float32. For the input data, two more arguments of type float32 must be provided representing the
    thresholds of quantizing argument from data type float32 to uint8. The final outputs contain the
    recurrent result in float32. It only supports quantization for Vanilla LSTM network.
    .. Note::
        This operator only supports forward propagation. DO NOT use it in training.

    Defined in ../src/operator/quantization/quantized_rnn.cc:L298

    Parameters
    ----------
    data : NDArray
        Input data.
    parameters : NDArray
        weight.
    state : NDArray
        initial hidden state of the RNN
    state_cell : NDArray
        initial cell state for LSTM networks (only for LSTM)
    data_scale : NDArray
        quantization scale of data.
    data_shift : NDArray
        quantization shift of data.
    state_size : int (non-negative), required
        size of the state for each layer
    num_layers : int (non-negative), required
        number of stacked layers
    bidirectional : boolean, optional, default=0
        whether to use bidirectional recurrent layers
    mode : {'gru', 'lstm', 'rnn_relu', 'rnn_tanh'}, required
        the type of RNN to compute
    p : float, optional, default=0
        drop rate of the dropout on the outputs of each RNN layer, except the last layer.
    state_outputs : boolean, optional, default=0
        Whether to have the states as symbol outputs.
    projection_size : int or None, optional, default='None'
        size of project size
    lstm_state_clip_min : double or None, optional, default=None
        Minimum clip value of LSTM states. This option must be used together with lstm_state_clip_max.
    lstm_state_clip_max : double or None, optional, default=None
        Maximum clip value of LSTM states. This option must be used together with lstm_state_clip_min.
    lstm_state_clip_nan : boolean, optional, default=0
        Whether to stop NaN from propagating in state by clipping it to min/max. If clipping range is not specified, this option is ignored.
    use_sequence_length : boolean, optional, default=0
        If set to true, this layer takes in an extra input parameter `sequence_length` to specify variable length sequence

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)