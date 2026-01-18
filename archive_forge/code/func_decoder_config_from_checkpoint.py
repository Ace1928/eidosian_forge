import argparse
from pathlib import Path
from typing import Dict, OrderedDict, Tuple
import torch
from audiocraft.models import MusicGen
from transformers import (
from transformers.models.musicgen_melody.configuration_musicgen_melody import MusicgenMelodyDecoderConfig
from transformers.models.musicgen_melody.feature_extraction_musicgen_melody import MusicgenMelodyFeatureExtractor
from transformers.models.musicgen_melody.modeling_musicgen_melody import (
from transformers.models.musicgen_melody.processing_musicgen_melody import MusicgenMelodyProcessor
from transformers.utils import logging
def decoder_config_from_checkpoint(checkpoint: str) -> MusicgenMelodyDecoderConfig:
    if checkpoint == 'facebook/musicgen-melody' or checkpoint == 'facebook/musicgen-stereo-melody':
        hidden_size = 1536
        num_hidden_layers = 48
        num_attention_heads = 24
    elif checkpoint == 'facebook/musicgen-melody-large' or checkpoint == 'facebook/musicgen-stereo-melody-large':
        hidden_size = 2048
        num_hidden_layers = 48
        num_attention_heads = 32
    else:
        raise ValueError(f"Checkpoint should be one of `['facebook/musicgen-melody', 'facebook/musicgen-melody-large']` for the mono checkpoints, or `['facebook/musicgen-stereo-melody', 'facebook/musicgen-stereo-melody-large']` for the stereo checkpoints, got {checkpoint}.")
    if 'stereo' in checkpoint:
        audio_channels = 2
        num_codebooks = 8
    else:
        audio_channels = 1
        num_codebooks = 4
    config = MusicgenMelodyDecoderConfig(hidden_size=hidden_size, ffn_dim=hidden_size * 4, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads, num_codebooks=num_codebooks, audio_channels=audio_channels)
    return config