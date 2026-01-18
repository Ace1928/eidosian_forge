import argparse
from pathlib import Path
import torch
from transformers import OPTConfig, OPTModel
from transformers.utils import logging
@torch.no_grad()
def convert_opt_checkpoint(checkpoint_path, pytorch_dump_folder_path, config=None):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    state_dict = load_checkpoint(checkpoint_path)
    if config is not None:
        config = OPTConfig.from_pretrained(config)
    else:
        config = OPTConfig()
    model = OPTModel(config).half().eval()
    model.load_state_dict(state_dict)
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)