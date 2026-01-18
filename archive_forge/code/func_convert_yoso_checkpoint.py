import argparse
import torch
from transformers import YosoConfig, YosoForMaskedLM
def convert_yoso_checkpoint(checkpoint_path, yoso_config_file, pytorch_dump_path):
    orig_state_dict = torch.load(checkpoint_path, map_location='cpu')['model_state_dict']
    config = YosoConfig.from_json_file(yoso_config_file)
    model = YosoForMaskedLM(config)
    new_state_dict = convert_checkpoint_helper(config.max_position_embeddings, orig_state_dict)
    print(model.load_state_dict(new_state_dict))
    model.eval()
    model.save_pretrained(pytorch_dump_path)
    print(f'Checkpoint successfuly converted. Model saved at {pytorch_dump_path}')