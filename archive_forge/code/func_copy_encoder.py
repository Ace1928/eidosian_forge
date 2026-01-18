import argparse
import torch
from clip import load
from transformers import CLIPConfig, CLIPModel
def copy_encoder(hf_encoder, pt_model):
    hf_encoder.embeddings.token_embedding.weight = pt_model.token_embedding.weight
    hf_encoder.embeddings.position_embedding.weight.data = pt_model.positional_embedding
    copy_linear(hf_encoder.final_layer_norm, pt_model.ln_final)
    copy_layers(hf_encoder.encoder.layers, pt_model.transformer.resblocks)