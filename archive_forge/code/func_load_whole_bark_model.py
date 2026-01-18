import argparse
import os
from pathlib import Path
import torch
from bark.generation import _load_model as _bark_load_model
from huggingface_hub import hf_hub_download
from transformers import EncodecConfig, EncodecModel, set_seed
from transformers.models.bark.configuration_bark import (
from transformers.models.bark.generation_configuration_bark import (
from transformers.models.bark.modeling_bark import BarkCoarseModel, BarkFineModel, BarkModel, BarkSemanticModel
from transformers.utils import logging
def load_whole_bark_model(semantic_path, coarse_path, fine_path, append_text, hub_path, folder_path):
    pytorch_dump_folder_path = os.path.join(folder_path, append_text)
    semanticConfig = BarkSemanticConfig.from_pretrained(os.path.join(semantic_path, 'config.json'))
    coarseAcousticConfig = BarkCoarseConfig.from_pretrained(os.path.join(coarse_path, 'config.json'))
    fineAcousticConfig = BarkFineConfig.from_pretrained(os.path.join(fine_path, 'config.json'))
    codecConfig = EncodecConfig.from_pretrained('facebook/encodec_24khz')
    semantic = BarkSemanticModel.from_pretrained(semantic_path)
    coarseAcoustic = BarkCoarseModel.from_pretrained(coarse_path)
    fineAcoustic = BarkFineModel.from_pretrained(fine_path)
    codec = EncodecModel.from_pretrained('facebook/encodec_24khz')
    bark_config = BarkConfig.from_sub_model_configs(semanticConfig, coarseAcousticConfig, fineAcousticConfig, codecConfig)
    bark_generation_config = BarkGenerationConfig.from_sub_model_configs(semantic.generation_config, coarseAcoustic.generation_config, fineAcoustic.generation_config)
    bark = BarkModel(bark_config)
    bark.semantic = semantic
    bark.coarse_acoustics = coarseAcoustic
    bark.fine_acoustics = fineAcoustic
    bark.codec_model = codec
    bark.generation_config = bark_generation_config
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    bark.save_pretrained(pytorch_dump_folder_path, repo_id=hub_path, push_to_hub=True)