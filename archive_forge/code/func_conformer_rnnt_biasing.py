import math
from typing import Dict, List, Optional, Tuple
import torch
from torchaudio.models import Conformer, RNNT
from torchaudio.models.rnnt import _Joiner, _Predictor, _TimeReduction, _Transcriber
def conformer_rnnt_biasing(*, input_dim: int, encoding_dim: int, time_reduction_stride: int, conformer_input_dim: int, conformer_ffn_dim: int, conformer_num_layers: int, conformer_num_heads: int, conformer_depthwise_conv_kernel_size: int, conformer_dropout: float, num_symbols: int, symbol_embedding_dim: int, num_lstm_layers: int, lstm_hidden_dim: int, lstm_layer_norm: int, lstm_layer_norm_epsilon: int, lstm_dropout: int, joiner_activation: str, attndim: int, biasing: bool, charlist: List[str], deepbiasing: bool, tcpsche: int, DBaverage: bool) -> RNNTBiasing:
    """Builds Conformer-based recurrent neural network transducer (RNN-T) model.

    Args:
        input_dim (int): dimension of input sequence frames passed to transcription network.
        encoding_dim (int): dimension of transcription- and prediction-network-generated encodings
            passed to joint network.
        time_reduction_stride (int): factor by which to reduce length of input sequence.
        conformer_input_dim (int): dimension of Conformer input.
        conformer_ffn_dim (int): hidden layer dimension of each Conformer layer's feedforward network.
        conformer_num_layers (int): number of Conformer layers to instantiate.
        conformer_num_heads (int): number of attention heads in each Conformer layer.
        conformer_depthwise_conv_kernel_size (int): kernel size of each Conformer layer's depthwise convolution layer.
        conformer_dropout (float): Conformer dropout probability.
        num_symbols (int): cardinality of set of target tokens.
        symbol_embedding_dim (int): dimension of each target token embedding.
        num_lstm_layers (int): number of LSTM layers to instantiate.
        lstm_hidden_dim (int): output dimension of each LSTM layer.
        lstm_layer_norm (bool): if ``True``, enables layer normalization for LSTM layers.
        lstm_layer_norm_epsilon (float): value of epsilon to use in LSTM layer normalization layers.
        lstm_dropout (float): LSTM dropout probability.
        joiner_activation (str): activation function to use in the joiner.
            Must be one of ("relu", "tanh"). (Default: "relu")
        attndim (int): TCPGen attention dimension
        biasing (bool): If true, use biasing, otherwise use standard RNN-T
        charlist (list): The list of word piece tokens in the same order as the output layer
        deepbiasing (bool): If true, use deep biasing by extracting the biasing vector
        tcpsche (int): The epoch at which TCPGen starts to train
        DBaverage (bool): If true, instead of TCPGen, use DBRNNT for biasing

        Returns:
            RNNT:
                Conformer RNN-T model with TCPGen-based biasing support.
    """
    encoder = _ConformerEncoder(input_dim=input_dim, output_dim=encoding_dim, time_reduction_stride=time_reduction_stride, conformer_input_dim=conformer_input_dim, conformer_ffn_dim=conformer_ffn_dim, conformer_num_layers=conformer_num_layers, conformer_num_heads=conformer_num_heads, conformer_depthwise_conv_kernel_size=conformer_depthwise_conv_kernel_size, conformer_dropout=conformer_dropout)
    predictor = _Predictor(num_symbols=num_symbols, output_dim=encoding_dim, symbol_embedding_dim=symbol_embedding_dim, num_lstm_layers=num_lstm_layers, lstm_hidden_dim=lstm_hidden_dim, lstm_layer_norm=lstm_layer_norm, lstm_layer_norm_epsilon=lstm_layer_norm_epsilon, lstm_dropout=lstm_dropout)
    joiner = _JoinerBiasing(encoding_dim, num_symbols, activation=joiner_activation, deepbiasing=deepbiasing, attndim=attndim, biasing=biasing)
    return RNNTBiasing(encoder, predictor, joiner, attndim, biasing, deepbiasing, symbol_embedding_dim, encoding_dim, charlist, encoding_dim, conformer_dropout, tcpsche, DBaverage)