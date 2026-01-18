import argparse
import torch
from transformers import ChineseCLIPConfig, ChineseCLIPModel
def copy_vision_model_and_projection(hf_model, pt_weights):
    hf_model.visual_projection.weight.data = pt_weights['visual.proj'].data.T
    copy_linear(hf_model.vision_model.pre_layrnorm, pt_weights, 'visual.ln_pre')
    copy_linear(hf_model.vision_model.post_layernorm, pt_weights, 'visual.ln_post')
    hf_model.vision_model.embeddings.patch_embedding.weight.data = pt_weights['visual.conv1.weight'].data
    hf_model.vision_model.embeddings.class_embedding.data = pt_weights['visual.class_embedding'].data
    hf_model.vision_model.embeddings.position_embedding.weight.data = pt_weights['visual.positional_embedding'].data
    copy_layers(hf_model.vision_model.encoder.layers, pt_weights, 'visual.transformer.resblocks')