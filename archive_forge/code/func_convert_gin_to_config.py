import argparse
import re
from flax.traverse_util import flatten_dict, unflatten_dict
from t5x import checkpoints
from transformers import SwitchTransformersConfig, SwitchTransformersForConditionalGeneration
from transformers.modeling_flax_pytorch_utils import load_flax_weights_in_pytorch_model
from transformers.utils import logging
def convert_gin_to_config(gin_file, num_experts):
    import regex as re
    with open(gin_file, 'r') as f:
        raw_gin = f.read()
    regex_match = re.findall('(.*) = ([0-9.]*)', raw_gin)
    args = {}
    for param, value in regex_match:
        if param in GIN_TO_CONFIG_MAPPING and value != '':
            args[GIN_TO_CONFIG_MAPPING[param]] = float(value) if '.' in value else int(value)
    activation = re.findall("(.*activations) = \\(\\'(.*)\\',\\)", raw_gin)[0]
    args[GIN_TO_CONFIG_MAPPING[activation[0]]] = str(activation[1])
    args['num_experts'] = num_experts
    config = SwitchTransformersConfig(**args)
    return config