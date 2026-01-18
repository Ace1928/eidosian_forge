import argparse
import os
import sys
import warnings
import flatdict
import torch
from transformers import FuyuConfig, FuyuForCausalLM, LlamaTokenizer
from transformers import FuyuForCausalLM, FuyuTokenizer
def convert_fuyu_checkpoint(pytorch_dump_folder_path, ada_lib_path, pt_model_path, safe_serialization=False):
    sys.path.insert(0, ada_lib_path)
    model_state_dict_base = torch.load(pt_model_path, map_location='cpu')
    state_dict = flatdict.FlatDict(model_state_dict_base['model'], '.')
    state_dict = rename_state_dict(state_dict)
    transformers_config = FuyuConfig()
    model = FuyuForCausalLM(transformers_config).to(torch.bfloat16)
    model.load_state_dict(state_dict)
    model.save_pretrained(pytorch_dump_folder_path, safe_serialization=safe_serialization)
    transformers_config.save_pretrained(pytorch_dump_folder_path)