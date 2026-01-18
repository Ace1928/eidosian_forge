import argparse
import json
import os
import re
import torch
from transformers import BloomConfig, BloomModel
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME
from transformers.utils import logging
def convert_bloom_checkpoint_to_pytorch(bloom_checkpoint_path, bloom_config_file, pytorch_dump_folder_path, shard_model, pretraining_tp):
    if bloom_config_file == '':
        config = BloomConfig()
    else:
        config = BloomConfig.from_json_file(bloom_config_file)
    if shard_model:
        file_names = os.listdir(bloom_checkpoint_path)
        file_names = sorted(filter(lambda s: s.startswith('layer') and 'model_00' in s, file_names))
        index_dict = {'weight_map': {}, 'metadata': {}}
        total_size = 0
        missing_keys = None
        config = BloomConfig()
        for j, file in enumerate(file_names):
            print('Processing file: {}'.format(file))
            tensors = None
            for i in range(pretraining_tp):
                f_name = file.replace('model_00', f'model_0{i}')
                temp = torch.load(os.path.join(bloom_checkpoint_path, f_name), map_location='cpu')
                keys = list(temp.keys())
                for key in keys:
                    temp[layer_name_mapping(key, file)] = temp.pop(key)
                if tensors is None:
                    tensors = temp
                else:
                    for key in tensors.keys():
                        if any((key.endswith(end) for end in WEIGHTS_TO_AVERAGE_ENDSWITH)):
                            tensors[key] += temp[key]
                        else:
                            cat_dim = 1 if any((text in key for text in WEIGHTS_WITH_ROW_PARALLELISM_CONTAIN)) else 0
                            tensors[key] = torch.cat([tensors[key], temp[key]], dim=cat_dim)
            for key in tensors.keys():
                if any((key.endswith(end) for end in WEIGHTS_TO_AVERAGE_ENDSWITH)):
                    tensors[key] = tensors[key] / pretraining_tp
            torch.save(tensors, os.path.join(pytorch_dump_folder_path, 'pytorch_model_{}-of-{}.bin'.format(str(j + 1).zfill(5), str(len(file_names)).zfill(5))))
            for key in tensors.keys():
                value = tensors[key]
                total_size += value.numel() * get_dtype_size(value.dtype)
                if key not in index_dict['weight_map']:
                    index_dict['weight_map'][key] = 'pytorch_model_{}-of-{}.bin'.format(str(j + 1).zfill(5), str(len(file_names)).zfill(5))
        config = BloomConfig()
        pytorch_config_dump_path = pytorch_dump_folder_path + '/' + CONFIG_NAME
        index_dict['metadata']['total_size'] = total_size
        with open(pytorch_config_dump_path, 'w', encoding='utf-8') as f:
            f.write(config.to_json_string())
        with open(os.path.join(pytorch_dump_folder_path, WEIGHTS_NAME + '.index.json'), 'w', encoding='utf-8') as f:
            json_config = json.dumps(index_dict, indent=2, sort_keys=True) + '\n'
            f.write(json_config)
    else:
        model = BloomModel(config)
        file_names = os.listdir(bloom_checkpoint_path)
        file_names = sorted(filter(lambda s: s.startswith('layer') and 'model_00' in s, file_names))
        missing_keys = None
        for i, file in enumerate(file_names):
            tensors = None
            for i in range(pretraining_tp):
                f_name = file.replace('model_00', f'model_0{i}')
                temp = torch.load(os.path.join(bloom_checkpoint_path, f_name), map_location='cpu')
                keys = list(temp.keys())
                for key in keys:
                    temp[layer_name_mapping(key, file)] = temp.pop(key)
                if tensors is None:
                    tensors = temp
                else:
                    for key in tensors.keys():
                        if any((key.endswith(end) for end in WEIGHTS_TO_AVERAGE_ENDSWITH)):
                            tensors[key] += temp[key]
                        else:
                            cat_dim = 1 if any((text in key for text in WEIGHTS_WITH_ROW_PARALLELISM_CONTAIN)) else 0
                            tensors[key] = torch.cat([tensors[key], temp[key]], dim=cat_dim)
            for key in tensors.keys():
                if any((key.endswith(end) for end in WEIGHTS_TO_AVERAGE_ENDSWITH)):
                    tensors[key] = tensors[key] / pretraining_tp
            other_keys = model.load_state_dict(tensors, strict=False)
            assert not other_keys.unexpected_keys, f'The keys {other_keys.unexpected_keys} are unexpected'
            if missing_keys is None:
                missing_keys = set(other_keys.missing_keys)
            else:
                missing_keys = missing_keys.intersection(set(other_keys.missing_keys))
        assert not missing_keys, f'The keys {missing_keys} are missing'
        os.makedirs(pytorch_dump_folder_path, exist_ok=True)
        pytorch_weights_dump_path = pytorch_dump_folder_path + '/' + WEIGHTS_NAME
        pytorch_config_dump_path = pytorch_dump_folder_path + '/' + CONFIG_NAME
        print(f'Save PyTorch model to {pytorch_weights_dump_path} with dtype {config.torch_dtype}')
        if config.torch_dtype is not None:
            model = model.to(config.torch_dtype)
        torch.save(model.state_dict(), pytorch_weights_dump_path)
        print(f'Save configuration file to {pytorch_config_dump_path}')
        with open(pytorch_config_dump_path, 'w', encoding='utf-8') as f:
            f.write(config.to_json_string())