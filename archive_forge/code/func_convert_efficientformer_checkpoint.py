import argparse
import re
from pathlib import Path
import requests
import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import (
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
def convert_efficientformer_checkpoint(checkpoint_path: Path, efficientformer_config_file: Path, pytorch_dump_path: Path, push_to_hub: bool):
    orig_state_dict = torch.load(checkpoint_path, map_location='cpu')['model']
    config = EfficientFormerConfig.from_json_file(efficientformer_config_file)
    model = EfficientFormerForImageClassificationWithTeacher(config)
    model_name = '_'.join(checkpoint_path.split('/')[-1].split('.')[0].split('_')[:-1])
    num_meta4D_last_stage = config.depths[-1] - config.num_meta3d_blocks + 1
    new_state_dict = convert_torch_checkpoint(orig_state_dict, num_meta4D_last_stage)
    model.load_state_dict(new_state_dict)
    model.eval()
    pillow_resamplings = {'bilinear': PILImageResampling.BILINEAR, 'bicubic': PILImageResampling.BICUBIC, 'nearest': PILImageResampling.NEAREST}
    image = prepare_img()
    image_size = 256
    crop_size = 224
    processor = EfficientFormerImageProcessor(size={'shortest_edge': image_size}, crop_size={'height': crop_size, 'width': crop_size}, resample=pillow_resamplings['bicubic'])
    pixel_values = processor(images=image, return_tensors='pt').pixel_values
    image_transforms = Compose([Resize(image_size, interpolation=pillow_resamplings['bicubic']), CenterCrop(crop_size), ToTensor(), Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
    original_pixel_values = image_transforms(image).unsqueeze(0)
    assert torch.allclose(original_pixel_values, pixel_values)
    outputs = model(pixel_values)
    logits = outputs.logits
    expected_shape = (1, 1000)
    if 'l1' in model_name:
        expected_logits = torch.Tensor([-0.1312, 0.4353, -1.0499, -0.5124, 0.4183, -0.6793, -1.3777, -0.0893, -0.7358, -2.4328])
        assert torch.allclose(logits[0, :10], expected_logits, atol=0.001)
        assert logits.shape == expected_shape
    elif 'l3' in model_name:
        expected_logits = torch.Tensor([-1.315, -1.5456, -1.2556, -0.8496, -0.7127, -0.7897, -0.9728, -0.3052, 0.3751, -0.3127])
        assert torch.allclose(logits[0, :10], expected_logits, atol=0.001)
        assert logits.shape == expected_shape
    elif 'l7' in model_name:
        expected_logits = torch.Tensor([-1.0283, -1.4131, -0.5644, -1.3115, -0.5785, -1.2049, -0.7528, 0.1992, -0.3822, -0.0878])
        assert logits.shape == expected_shape
    else:
        raise ValueError(f'Unknown model checkpoint: {checkpoint_path}. Supported version of efficientformer are l1, l3 and l7')
    Path(pytorch_dump_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_path)
    print(f'Checkpoint successfuly converted. Model saved at {pytorch_dump_path}')
    processor.save_pretrained(pytorch_dump_path)
    print(f'Processor successfuly saved at {pytorch_dump_path}')
    if push_to_hub:
        print('Pushing model to the hub...')
        model.push_to_hub(repo_id=f'Bearnardd/{pytorch_dump_path}', commit_message='Add model', use_temp_dir=True)
        processor.push_to_hub(repo_id=f'Bearnardd/{pytorch_dump_path}', commit_message='Add image processor', use_temp_dir=True)