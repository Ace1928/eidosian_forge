import argparse
import torch
from clip import load
from transformers import CLIPConfig, CLIPModel
def copy_vison_model_and_projection(hf_model, pt_model):
    hf_model.visual_projection.weight.data = pt_model.visual.proj.data.T
    copy_linear(hf_model.vision_model.pre_layrnorm, pt_model.visual.ln_pre)
    copy_linear(hf_model.vision_model.post_layernorm, pt_model.visual.ln_post)
    hf_model.vision_model.embeddings.patch_embedding.weight.data = pt_model.visual.conv1.weight.data
    hf_model.vision_model.embeddings.class_embedding = pt_model.visual.class_embedding
    hf_model.vision_model.embeddings.position_embedding.weight.data = pt_model.visual.positional_embedding.data
    copy_layers(hf_model.vision_model.encoder.layers, pt_model.visual.transformer.resblocks)