from typing import List, Optional, Tuple
import torch
from torchaudio.models import Wav2Vec2Model
from torchaudio.models.emformer import Emformer
from torchaudio.models.rnnt import _TimeReduction
def emformer_hubert_model(extractor_input_dim: int, extractor_output_dim: int, extractor_use_bias: bool, extractor_stride: int, encoder_input_dim: int, encoder_output_dim: int, encoder_num_heads: int, encoder_ffn_dim: int, encoder_num_layers: int, encoder_segment_length: int, encoder_left_context_length: int, encoder_right_context_length: int, encoder_dropout: float, encoder_activation: str, encoder_max_memory_size: int, encoder_weight_init_scale_strategy: Optional[str], encoder_tanh_on_mem: bool, aux_num_out: Optional[int]) -> Wav2Vec2Model:
    """Build a custom Emformer HuBERT model.

    Args:
        extractor_input_dim (int): The input dimension for feature extractor.
        extractor_output_dim (int): The output dimension after feature extractor.
        extractor_use_bias (bool): If ``True``, enable bias parameter in the linear layer of feature extractor.
        extractor_stride (int): Number of frames to merge for the output frame in feature extractor.
        encoder_input_dim (int): The input dimension for Emformer layer.
        encoder_output_dim (int): The output dimension after EmformerEncoder.
        encoder_num_heads (int): Number of attention heads in each Emformer layer.
        encoder_ffn_dim (int): Hidden layer dimension of feedforward network in Emformer.
        encoder_num_layers (int): Number of Emformer layers to instantiate.
        encoder_segment_length (int): Length of each input segment.
        encoder_left_context_length (int): Length of left context.
        encoder_right_context_length (int): Length of right context.
        encoder_dropout (float): Dropout probability.
        encoder_activation (str): Activation function to use in each Emformer layer's
            feedforward network. Must be one of ("relu", "gelu", "silu").
        encoder_max_memory_size (int): Maximum number of memory elements to use.
        encoder_weight_init_scale_strategy (str or None): Per-layer weight initialization scaling
            strategy. Must be one of ("depthwise", "constant", ``None``).
        encoder_tanh_on_mem (bool): If ``True``, applies tanh to memory elements.
        aux_num_out (int or None):
            When provided, attach an extra linear layer on top of encoder, which can be
            used for fine-tuning.

    Returns:
        Wav2Vec2Model:
            The resulting :py:class:`torchaudio.models.Wav2Vec2Model` model
            with a :py:class:`torchaudio.models.Emformer` encoder.
    """
    feature_extractor = _get_emformer_feature_extractor(extractor_input_dim, extractor_output_dim, extractor_use_bias, extractor_stride)
    emformer = _get_emformer_encoder(encoder_input_dim, encoder_output_dim, encoder_num_heads, encoder_ffn_dim, encoder_num_layers, encoder_segment_length, encoder_left_context_length, encoder_right_context_length, encoder_dropout, encoder_activation, encoder_max_memory_size, encoder_weight_init_scale_strategy, encoder_tanh_on_mem)
    aux = None
    if aux_num_out is not None:
        aux = torch.nn.Linear(in_features=encoder_output_dim, out_features=aux_num_out)
    return Wav2Vec2Model(feature_extractor, emformer, aux)