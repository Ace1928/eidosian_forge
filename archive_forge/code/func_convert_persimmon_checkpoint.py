import argparse
import os
import warnings
import flatdict
import torch
from transformers import LlamaTokenizer, PersimmonConfig, PersimmonForCausalLM
from transformers import PersimmonForCausalLM, PersimmonTokenizer
def convert_persimmon_checkpoint(pytorch_dump_folder_path, ada_lib_path, pt_model_path, safe_serialization=False):
    import sys
    sys.path.insert(0, ada_lib_path)
    model_state_dict_base = torch.load(pt_model_path, map_location='cpu')
    state_dict = flatdict.FlatDict(model_state_dict_base['model'], '.')
    state_dict = rename_state_dict(state_dict)
    transformers_config = PersimmonConfig()
    model = PersimmonForCausalLM(transformers_config, eos_token_id=71013, bos_token_id=71013).to(torch.bfloat16)
    model.load_state_dict(state_dict)
    model.save_pretrained(pytorch_dump_folder_path, safe_serialization=safe_serialization)
    transformers_config.save_pretrained(pytorch_dump_folder_path)