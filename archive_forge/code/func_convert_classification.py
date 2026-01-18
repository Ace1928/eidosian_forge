import argparse
import torch
from transformers import (
def convert_classification(base_model_name, hf_config, downstream_dict):
    model = WavLMForSequenceClassification.from_pretrained(base_model_name, config=hf_config)
    model.projector.weight.data = downstream_dict['projector.weight']
    model.projector.bias.data = downstream_dict['projector.bias']
    model.classifier.weight.data = downstream_dict['model.post_net.linear.weight']
    model.classifier.bias.data = downstream_dict['model.post_net.linear.bias']
    return model