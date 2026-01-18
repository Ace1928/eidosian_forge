import argparse
import torch
from transformers import OpenAIGPTConfig, OpenAIGPTModel, load_tf_weights_in_openai_gpt
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging
def convert_openai_checkpoint_to_pytorch(openai_checkpoint_folder_path, openai_config_file, pytorch_dump_folder_path):
    if openai_config_file == '':
        config = OpenAIGPTConfig()
    else:
        config = OpenAIGPTConfig.from_json_file(openai_config_file)
    model = OpenAIGPTModel(config)
    load_tf_weights_in_openai_gpt(model, config, openai_checkpoint_folder_path)
    pytorch_weights_dump_path = pytorch_dump_folder_path + '/' + WEIGHTS_NAME
    pytorch_config_dump_path = pytorch_dump_folder_path + '/' + CONFIG_NAME
    print(f'Save PyTorch model to {pytorch_weights_dump_path}')
    torch.save(model.state_dict(), pytorch_weights_dump_path)
    print(f'Save configuration file to {pytorch_config_dump_path}')
    with open(pytorch_config_dump_path, 'w', encoding='utf-8') as f:
        f.write(config.to_json_string())