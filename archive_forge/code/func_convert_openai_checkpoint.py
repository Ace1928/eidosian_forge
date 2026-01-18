import argparse
import json
import os
from pathlib import Path
import requests
import torch
from transformers import JukeboxConfig, JukeboxModel
from transformers.utils import logging
@torch.no_grad()
def convert_openai_checkpoint(model_name=None, pytorch_dump_folder_path=None):
    """
    Copy/paste/tweak model's weights to our Jukebox structure.
    """
    for file in MODEL_MAPPING[model_name]:
        if not os.path.isfile(f'{pytorch_dump_folder_path}/{file.split('/')[-1]}'):
            r = requests.get(f'{PREFIX}{file}', allow_redirects=True)
            os.makedirs(f'{pytorch_dump_folder_path}/', exist_ok=True)
            open(f'{pytorch_dump_folder_path}/{file.split('/')[-1]}', 'wb').write(r.content)
    model_to_convert = MODEL_MAPPING[model_name.split('/')[-1]]
    config = JukeboxConfig.from_pretrained(model_name)
    model = JukeboxModel(config)
    weight_dict = []
    mapping = {}
    for i, dict_name in enumerate(model_to_convert):
        old_dic = torch.load(f'{pytorch_dump_folder_path}/{dict_name.split('/')[-1]}')['model']
        new_dic = {}
        for k in old_dic.keys():
            if k.endswith('.b'):
                new_dic[k.replace('b', 'bias')] = old_dic[k]
            elif k.endswith('.w'):
                new_dic[k.replace('w', 'weight')] = old_dic[k]
            elif 'level_2' not in dict_name and 'cond.model.' in k:
                new_dic[k.replace('.blocks.', '.model.')] = old_dic[k]
            else:
                new_dic[k] = old_dic[k]
        key_prefix = 'vqvae' if i == 0 else f'priors.{3 - i}'
        new_dic = fix_jukebox_keys(new_dic, model.state_dict(), key_prefix, mapping)
        weight_dict.append(new_dic)
    vqvae_state_dict = weight_dict.pop(0)
    model.vqvae.load_state_dict(vqvae_state_dict)
    for i in range(len(weight_dict)):
        model.priors[i].load_state_dict(weight_dict[2 - i])
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    with open(f'{pytorch_dump_folder_path}/mapping.json', 'w') as txtfile:
        json.dump(mapping, txtfile)
    print(f'Saving model {model_name} to {pytorch_dump_folder_path}')
    model.save_pretrained(pytorch_dump_folder_path)
    return weight_dict