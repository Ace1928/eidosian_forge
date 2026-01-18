import argparse
import torch
from transformers import (
def convert_diarization(base_model_name, hf_config, downstream_dict):
    model = WavLMForAudioFrameClassification.from_pretrained(base_model_name, config=hf_config)
    model.classifier.weight.data = downstream_dict['model.linear.weight']
    model.classifier.bias.data = downstream_dict['model.linear.bias']
    return model