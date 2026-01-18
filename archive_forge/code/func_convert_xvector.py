import argparse
import torch
from transformers import (
def convert_xvector(base_model_name, hf_config, downstream_dict):
    model = WavLMForXVector.from_pretrained(base_model_name, config=hf_config)
    model.projector.weight.data = downstream_dict['connector.weight']
    model.projector.bias.data = downstream_dict['connector.bias']
    for i, kernel_size in enumerate(hf_config.tdnn_kernel):
        model.tdnn[i].kernel.weight.data = downstream_dict[f'model.framelevel_feature_extractor.module.{i}.kernel.weight']
        model.tdnn[i].kernel.bias.data = downstream_dict[f'model.framelevel_feature_extractor.module.{i}.kernel.bias']
    model.feature_extractor.weight.data = downstream_dict['model.utterancelevel_feature_extractor.linear1.weight']
    model.feature_extractor.bias.data = downstream_dict['model.utterancelevel_feature_extractor.linear1.bias']
    model.classifier.weight.data = downstream_dict['model.utterancelevel_feature_extractor.linear2.weight']
    model.classifier.bias.data = downstream_dict['model.utterancelevel_feature_extractor.linear2.bias']
    model.objective.weight.data = downstream_dict['objective.W']
    return model