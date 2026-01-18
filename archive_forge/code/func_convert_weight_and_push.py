import argparse
import json
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List
import torch
import torch.nn as nn
from huggingface_hub import cached_download, hf_hub_download
from torch import Tensor
from transformers import AutoImageProcessor, VanConfig, VanForImageClassification
from transformers.models.deprecated.van.modeling_van import VanLayerScaling
from transformers.utils import logging
def convert_weight_and_push(name: str, config: VanConfig, checkpoint: str, from_model: nn.Module, save_directory: Path, push_to_hub: bool=True):
    print(f'Downloading weights for {name}...')
    checkpoint_path = cached_download(checkpoint)
    print(f'Converting {name}...')
    from_state_dict = torch.load(checkpoint_path)['state_dict']
    from_model.load_state_dict(from_state_dict)
    from_model.eval()
    with torch.no_grad():
        our_model = VanForImageClassification(config).eval()
        module_transfer = ModuleTransfer(src=from_model, dest=our_model)
        x = torch.randn((1, 3, 224, 224))
        module_transfer(x)
        our_model = copy_parameters(from_model, our_model)
    if not torch.allclose(from_model(x), our_model(x).logits):
        raise ValueError("The model logits don't match the original one.")
    checkpoint_name = name
    print(checkpoint_name)
    if push_to_hub:
        our_model.push_to_hub(repo_path_or_name=save_directory / checkpoint_name, commit_message='Add model', use_temp_dir=True)
        image_processor = AutoImageProcessor.from_pretrained('facebook/convnext-base-224-22k-1k')
        image_processor.push_to_hub(repo_path_or_name=save_directory / checkpoint_name, commit_message='Add image processor', use_temp_dir=True)
        print(f'Pushed {checkpoint_name}')