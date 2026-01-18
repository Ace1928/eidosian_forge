import argparse
import os
import torch
from transformers import FlavaImageCodebook, FlavaImageCodebookConfig
@torch.no_grad()
def convert_dalle_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None, save_checkpoint=True):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    from dall_e import Encoder
    encoder = Encoder()
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path)
    else:
        ckpt = torch.hub.load_state_dict_from_url(checkpoint_path)
    if isinstance(ckpt, Encoder):
        ckpt = ckpt.state_dict()
    encoder.load_state_dict(ckpt)
    if config_path is not None:
        config = FlavaImageCodebookConfig.from_pretrained(config_path)
    else:
        config = FlavaImageCodebookConfig()
    hf_model = FlavaImageCodebook(config).eval()
    state_dict = encoder.state_dict()
    hf_state_dict = upgrade_state_dict(state_dict)
    hf_model.load_state_dict(hf_state_dict)
    hf_state_dict = hf_model.state_dict()
    hf_count = count_parameters(hf_state_dict)
    state_dict_count = count_parameters(state_dict)
    assert torch.allclose(hf_count, state_dict_count, atol=0.001)
    if save_checkpoint:
        hf_model.save_pretrained(pytorch_dump_folder_path)
    else:
        return hf_state_dict