import argparse
import torch
from unilm.wavlm.WavLM import WavLM as WavLMOrig
from unilm.wavlm.WavLM import WavLMConfig as WavLMConfigOrig
from transformers import WavLMConfig, WavLMModel, logging
@torch.no_grad()
def convert_wavlm_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None):
    checkpoint = torch.load(checkpoint_path)
    cfg = WavLMConfigOrig(checkpoint['cfg'])
    model = WavLMOrig(cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    if config_path is not None:
        config = WavLMConfig.from_pretrained(config_path)
    else:
        config = WavLMConfig()
    hf_wavlm = WavLMModel(config)
    recursively_load_weights(model, hf_wavlm)
    hf_wavlm.save_pretrained(pytorch_dump_folder_path)