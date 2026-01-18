import argparse
from collections import OrderedDict
from pathlib import Path
import requests
import torch
from PIL import Image
from transformers import GLPNConfig, GLPNForDepthEstimation, GLPNImageProcessor
from transformers.utils import logging
@torch.no_grad()
def convert_glpn_checkpoint(checkpoint_path, pytorch_dump_folder_path, push_to_hub=False, model_name=None):
    """
    Copy/paste/tweak model's weights to our GLPN structure.
    """
    config = GLPNConfig(hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=64, depths=[3, 8, 27, 3])
    image_processor = GLPNImageProcessor()
    image = prepare_img()
    pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
    logger.info('Converting model...')
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = rename_keys(state_dict)
    read_in_k_v(state_dict, config)
    model = GLPNForDepthEstimation(config)
    model.load_state_dict(state_dict)
    model.eval()
    outputs = model(pixel_values)
    predicted_depth = outputs.predicted_depth
    if model_name is not None:
        if 'nyu' in model_name:
            expected_slice = torch.tensor([[4.4147, 4.0873, 4.0673], [3.789, 3.2881, 3.1525], [3.7674, 3.5423, 3.4913]])
        elif 'kitti' in model_name:
            expected_slice = torch.tensor([[3.4291, 2.7865, 2.5151], [3.2841, 2.7021, 2.3502], [3.1147, 2.4625, 2.2481]])
        else:
            raise ValueError(f'Unknown model name: {model_name}')
        expected_shape = torch.Size([1, 480, 640])
        assert predicted_depth.shape == expected_shape
        assert torch.allclose(predicted_depth[0, :3, :3], expected_slice, atol=0.0001)
        print('Looks ok!')
    if push_to_hub:
        logger.info('Pushing model and image processor to the hub...')
        model.push_to_hub(repo_path_or_name=Path(pytorch_dump_folder_path, model_name), organization='nielsr', commit_message='Add model', use_temp_dir=True)
        image_processor.push_to_hub(repo_path_or_name=Path(pytorch_dump_folder_path, model_name), organization='nielsr', commit_message='Add image processor', use_temp_dir=True)