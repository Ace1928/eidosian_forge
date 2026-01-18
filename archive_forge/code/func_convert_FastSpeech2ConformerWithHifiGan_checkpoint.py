import argparse
import torch
from transformers import (
from .convert_fastspeech2_conformer_original_pytorch_checkpoint_to_pytorch import (
from .convert_hifigan import load_weights, remap_hifigan_yaml_config
def convert_FastSpeech2ConformerWithHifiGan_checkpoint(checkpoint_path, yaml_config_path, pytorch_dump_folder_path, repo_id=None):
    model_params, *_ = remap_model_yaml_config(yaml_config_path)
    model_config = FastSpeech2ConformerConfig(**model_params)
    model = FastSpeech2ConformerModel(model_config)
    espnet_checkpoint = torch.load(checkpoint_path)
    hf_compatible_state_dict = convert_espnet_state_dict_to_hf(espnet_checkpoint)
    model.load_state_dict(hf_compatible_state_dict)
    config_kwargs = remap_hifigan_yaml_config(yaml_config_path)
    vocoder_config = FastSpeech2ConformerHifiGanConfig(**config_kwargs)
    vocoder = FastSpeech2ConformerHifiGan(vocoder_config)
    load_weights(espnet_checkpoint, vocoder, vocoder_config)
    config = FastSpeech2ConformerWithHifiGanConfig.from_sub_model_configs(model_config, vocoder_config)
    with_hifigan_model = FastSpeech2ConformerWithHifiGan(config)
    with_hifigan_model.model = model
    with_hifigan_model.vocoder = vocoder
    with_hifigan_model.save_pretrained(pytorch_dump_folder_path)
    if repo_id:
        print('Pushing to the hub...')
        with_hifigan_model.push_to_hub(repo_id)