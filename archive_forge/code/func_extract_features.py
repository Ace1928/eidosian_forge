from typing import List, Optional, Tuple
import torch
from torch import nn, Tensor
from torchaudio._internal import load_state_dict_from_url
from torchaudio.models import wav2vec2_model, Wav2Vec2Model, wavlm_model
@torch.jit.export
def extract_features(self, waveforms: Tensor, lengths: Optional[Tensor]=None, num_layers: Optional[int]=None) -> Tuple[List[Tensor], Optional[Tensor]]:
    if self.normalize_waveform:
        waveforms = nn.functional.layer_norm(waveforms, waveforms.shape)
    return self.model.extract_features(waveforms, lengths, num_layers)