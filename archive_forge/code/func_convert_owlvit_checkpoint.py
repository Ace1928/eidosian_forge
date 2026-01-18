import argparse
import collections
import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
from clip.model import CLIP
from flax.training import checkpoints
from huggingface_hub import Repository
from transformers import (
@torch.no_grad()
def convert_owlvit_checkpoint(pt_backbone, flax_params, attn_params, pytorch_dump_folder_path, config_path=None):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    repo = Repository(pytorch_dump_folder_path, clone_from=f'google/{pytorch_dump_folder_path}')
    repo.git_pull()
    if config_path is not None:
        config = OwlViTConfig.from_pretrained(config_path)
    else:
        config = OwlViTConfig()
    hf_backbone = OwlViTModel(config).eval()
    hf_model = OwlViTForObjectDetection(config).eval()
    copy_text_model_and_projection(hf_backbone, pt_backbone)
    copy_vision_model_and_projection(hf_backbone, pt_backbone)
    hf_backbone.logit_scale = pt_backbone.logit_scale
    copy_flax_attn_params(hf_backbone, attn_params)
    hf_model.owlvit = hf_backbone
    copy_class_merge_token(hf_model, flax_params)
    copy_class_box_heads(hf_model, flax_params)
    hf_model.save_pretrained(repo.local_dir)
    image_processor = OwlViTImageProcessor(size=config.vision_config.image_size, crop_size=config.vision_config.image_size)
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32', pad_token='!', model_max_length=16)
    processor = OwlViTProcessor(image_processor=image_processor, tokenizer=tokenizer)
    image_processor.save_pretrained(repo.local_dir)
    processor.save_pretrained(repo.local_dir)
    repo.git_add()
    repo.git_commit('Upload model and processor')
    repo.git_push()