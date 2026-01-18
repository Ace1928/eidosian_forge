import argparse
import re
from flax.traverse_util import flatten_dict, unflatten_dict
from t5x import checkpoints
from transformers import SwitchTransformersConfig, SwitchTransformersForConditionalGeneration
from transformers.modeling_flax_pytorch_utils import load_flax_weights_in_pytorch_model
from transformers.utils import logging
def convert_flax_checkpoint_to_pytorch(flax_checkpoint_path, config_file, gin_file=None, pytorch_dump_path='./', num_experts=8):
    print(f'Loading flax weights from : {flax_checkpoint_path}')
    flax_params = checkpoints.load_t5x_checkpoint(flax_checkpoint_path)
    if gin_file is not None:
        config = convert_gin_to_config(gin_file, num_experts)
    else:
        config = SwitchTransformersConfig.from_pretrained(config_file)
    pt_model = SwitchTransformersForConditionalGeneration(config)
    flax_params = flax_params['target']
    flax_params = flatten_dict(flax_params, sep='/')
    flax_params = rename_keys(flax_params)
    flax_params = unflatten_dict(flax_params, sep='/')
    load_flax_weights_in_pytorch_model(pt_model, flax_params)
    print(f'Save PyTorch model to {pytorch_dump_path}')
    pt_model.save_pretrained(pytorch_dump_path)