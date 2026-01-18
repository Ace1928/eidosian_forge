import argparse
import bros  # original repo
import torch
from transformers import BrosConfig, BrosModel, BrosProcessor
from transformers.utils import logging
def convert_bros_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    original_model = bros.BrosModel.from_pretrained(model_name).eval()
    bros_config = get_configs(model_name)
    model = BrosModel.from_pretrained(model_name, config=bros_config)
    model.eval()
    state_dict = original_model.state_dict()
    new_state_dict = convert_state_dict(state_dict, model)
    model.load_state_dict(new_state_dict)
    bbox = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4396, 0.672, 0.4659, 0.672, 0.4659, 0.685, 0.4396, 0.685], [0.4698, 0.672, 0.4843, 0.672, 0.4843, 0.685, 0.4698, 0.685], [0.4698, 0.672, 0.4843, 0.672, 0.4843, 0.685, 0.4698, 0.685], [0.2047, 0.687, 0.273, 0.687, 0.273, 0.7, 0.2047, 0.7], [0.2047, 0.687, 0.273, 0.687, 0.273, 0.7, 0.2047, 0.7], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]])
    processor = BrosProcessor.from_pretrained(model_name)
    encoding = processor('His name is Rocco.', return_tensors='pt')
    encoding['bbox'] = bbox
    original_hidden_states = original_model(**encoding).last_hidden_state
    last_hidden_states = model(**encoding).last_hidden_state
    assert torch.allclose(original_hidden_states, last_hidden_states, atol=0.0001)
    if pytorch_dump_folder_path is not None:
        print(f'Saving model and processor to {pytorch_dump_folder_path}')
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        model.push_to_hub('jinho8345/' + model_name.split('/')[-1], commit_message='Update model')
        processor.push_to_hub('jinho8345/' + model_name.split('/')[-1], commit_message='Update model')