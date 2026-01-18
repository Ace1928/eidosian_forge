import argparse
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation, Dinov2Config, DPTImageProcessor
from transformers.utils import logging
@torch.no_grad()
def convert_dpt_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub, verify_logits):
    """
    Copy/paste/tweak model's weights to our DPT structure.
    """
    config = get_dpt_config(model_name)
    model_name_to_filename = {'depth-anything-small': 'depth_anything_vits14.pth', 'depth-anything-base': 'depth_anything_vitb14.pth', 'depth-anything-large': 'depth_anything_vitl14.pth'}
    filename = model_name_to_filename[model_name]
    filepath = hf_hub_download(repo_id='LiheYoung/Depth-Anything', filename=f'checkpoints/{filename}', repo_type='space')
    state_dict = torch.load(filepath, map_location='cpu')
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config)
    model = DepthAnythingForDepthEstimation(config)
    model.load_state_dict(state_dict)
    model.eval()
    processor = DPTImageProcessor(do_resize=True, size={'height': 518, 'width': 518}, ensure_multiple_of=14, keep_aspect_ratio=True, do_rescale=True, do_normalize=True, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    pixel_values = processor(image, return_tensors='pt').pixel_values
    with torch.no_grad():
        outputs = model(pixel_values)
        predicted_depth = outputs.predicted_depth
    print('Shape of predicted depth:', predicted_depth.shape)
    print('First values:', predicted_depth[0, :3, :3])
    if verify_logits:
        expected_shape = torch.Size([1, 518, 686])
        if model_name == 'depth-anything-small':
            expected_slice = torch.tensor([[8.8204, 8.6468, 8.6195], [8.3313, 8.6027, 8.7526], [8.6526, 8.6866, 8.7453]])
        elif model_name == 'depth-anything-base':
            expected_slice = torch.tensor([[26.3997, 26.3004, 26.3928], [26.226, 26.2092, 26.3427], [26.0719, 26.0483, 26.1254]])
        elif model_name == 'depth-anything-large':
            expected_slice = torch.tensor([[87.9968, 87.7493, 88.2704], [87.1927, 87.6611, 87.364], [86.7789, 86.9469, 86.7991]])
        else:
            raise ValueError('Not supported')
        assert predicted_depth.shape == torch.Size(expected_shape)
        assert torch.allclose(predicted_depth[0, :3, :3], expected_slice, atol=1e-06)
        print('Looks ok!')
    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f'Saving model and processor to {pytorch_dump_folder_path}')
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        print('Pushing model and processor to hub...')
        model.push_to_hub(repo_id=f'LiheYoung/{model_name}-hf')
        processor.push_to_hub(repo_id=f'LiheYoung/{model_name}-hf')