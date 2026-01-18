import argparse
import torch
from transformers import GPT2Config, GPT2Model, load_tf_weights_in_gpt2
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging
def convert_gpt2_checkpoint_to_pytorch(gpt2_checkpoint_path, gpt2_config_file, pytorch_dump_folder_path):
    if gpt2_config_file == '':
        config = GPT2Config()
    else:
        config = GPT2Config.from_json_file(gpt2_config_file)
    model = GPT2Model(config)
    load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path)
    pytorch_weights_dump_path = pytorch_dump_folder_path + '/' + WEIGHTS_NAME
    pytorch_config_dump_path = pytorch_dump_folder_path + '/' + CONFIG_NAME
    print(f'Save PyTorch model to {pytorch_weights_dump_path}')
    torch.save(model.state_dict(), pytorch_weights_dump_path)
    print(f'Save configuration file to {pytorch_config_dump_path}')
    with open(pytorch_config_dump_path, 'w', encoding='utf-8') as f:
        f.write(config.to_json_string())