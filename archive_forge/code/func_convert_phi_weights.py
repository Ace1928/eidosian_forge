import argparse
import gc
import os
import safetensors
import torch
from huggingface_hub import hf_hub_download
from transformers import PhiConfig, PhiForCausalLM
def convert_phi_weights(model_name, checkpoint_path, pytorch_dump_folder_path, use_cuda, save_weights_directly, _MODELS):
    _MODELS = _MODELS if model_name not in _MODELS.keys() else {model_name: _MODELS.get(model_name)}
    device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'
    for model_name, model_url in _MODELS.items():
        converted_checkpoint = {}
        model_checkpoint = {}
        for model_each_url in model_url:
            model_path = os.path.join(checkpoint_path, model_name + '_' + model_each_url.split('/')[-1])
            if not os.path.exists(model_path):
                print(f'\n{model_name} was not found! Downloading it to {model_path}')
                _download(url=model_each_url, root=model_path)
            if model_path.endswith('safetensors'):
                loaded_weights = safetensors.torch.load_file(model_path, device=device)
            else:
                loaded_weights = torch.load(model_path, map_location=device)
            model_checkpoint.update(**loaded_weights)
        model_type = model_name.split('/')[1]
        config = PhiConfig()
        if model_type == 'phi-2':
            config.hidden_size = 2560
            config.intermediate_size = 10240
            config.num_hidden_layers = 32
            config.resid_pdrop = 0.1
            config.partial_rotary_factor = 0.4
            config.num_hidden_layers = 32
            config.torch_dtype = 'float16'
        converted_checkpoint.update(**convert_weights(model_checkpoint, PHI_MAPPING, config))
        if save_weights_directly:
            save_weights_path = os.path.join(pytorch_dump_folder_path, model_type + '_pytorch_model.bin')
            torch.save(converted_checkpoint, save_weights_path)
            print(f'Model weights saved at {save_weights_path}!')
        else:
            model = PhiForCausalLM(config).to(device)
            model.load_state_dict(converted_checkpoint, strict=True)
            save_model_path = os.path.join(pytorch_dump_folder_path, model_type)
            model.save_pretrained(save_model_path)
            print(f'Model saved at {save_model_path}!')
            del config, model
        del model_checkpoint, converted_checkpoint
        if use_cuda:
            torch.cuda.empty_cache()
        gc.collect()