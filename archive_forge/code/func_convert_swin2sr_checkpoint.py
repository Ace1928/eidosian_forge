import argparse
import requests
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution, Swin2SRImageProcessor
def convert_swin2sr_checkpoint(checkpoint_url, pytorch_dump_folder_path, push_to_hub):
    config = get_config(checkpoint_url)
    model = Swin2SRForImageSuperResolution(config)
    model.eval()
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu')
    new_state_dict = convert_state_dict(state_dict, config)
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    if len(missing_keys) > 0:
        raise ValueError('Missing keys when converting: {}'.format(missing_keys))
    for key in unexpected_keys:
        if not ('relative_position_index' in key or 'relative_coords_table' in key or 'self_mask' in key):
            raise ValueError(f'Unexpected key {key} in state_dict')
    url = 'https://github.com/mv-lab/swin2sr/blob/main/testsets/real-inputs/shanghai.jpg?raw=true'
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    processor = Swin2SRImageProcessor()
    image_size = 126 if 'Jpeg' in checkpoint_url else 256
    transforms = Compose([Resize((image_size, image_size)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    pixel_values = transforms(image).unsqueeze(0)
    if config.num_channels == 1:
        pixel_values = pixel_values[:, 0, :, :].unsqueeze(1)
    outputs = model(pixel_values)
    if 'Swin2SR_ClassicalSR_X2_64' in checkpoint_url:
        expected_shape = torch.Size([1, 3, 512, 512])
        expected_slice = torch.tensor([[-0.7087, -0.7138, -0.6721], [-0.834, -0.8095, -0.7298], [-0.9149, -0.8414, -0.794]])
    elif 'Swin2SR_ClassicalSR_X4_64' in checkpoint_url:
        expected_shape = torch.Size([1, 3, 1024, 1024])
        expected_slice = torch.tensor([[-0.7775, -0.8105, -0.8933], [-0.7764, -0.8356, -0.9225], [-0.7976, -0.8686, -0.9579]])
    elif 'Swin2SR_CompressedSR_X4_48' in checkpoint_url:
        expected_shape = torch.Size([1, 3, 1024, 1024])
        expected_slice = torch.tensor([[-0.8035, -0.7504, -0.7491], [-0.8538, -0.8124, -0.7782], [-0.8804, -0.8651, -0.8493]])
    elif 'Swin2SR_Lightweight_X2_64' in checkpoint_url:
        expected_shape = torch.Size([1, 3, 512, 512])
        expected_slice = torch.tensor([[-0.7669, -0.8662, -0.8767], [-0.881, -0.9962, -0.982], [-0.934, -1.0322, -1.1149]])
    elif 'Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR' in checkpoint_url:
        expected_shape = torch.Size([1, 3, 1024, 1024])
        expected_slice = torch.tensor([[-0.5238, -0.5557, -0.6321], [-0.6016, -0.5903, -0.6391], [-0.6244, -0.6334, -0.6889]])
    assert outputs.reconstruction.shape == expected_shape, f'Shape of reconstruction should be {expected_shape}, but is {outputs.reconstruction.shape}'
    assert torch.allclose(outputs.reconstruction[0, 0, :3, :3], expected_slice, atol=0.001)
    print('Looks ok!')
    url_to_name = {'https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_ClassicalSR_X2_64.pth': 'swin2SR-classical-sr-x2-64', 'https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_ClassicalSR_X4_64.pth': 'swin2SR-classical-sr-x4-64', 'https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_CompressedSR_X4_48.pth': 'swin2SR-compressed-sr-x4-48', 'https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_Lightweight_X2_64.pth': 'swin2SR-lightweight-x2-64', 'https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth': 'swin2SR-realworld-sr-x4-64-bsrgan-psnr'}
    model_name = url_to_name[checkpoint_url]
    if pytorch_dump_folder_path is not None:
        print(f'Saving model {model_name} to {pytorch_dump_folder_path}')
        model.save_pretrained(pytorch_dump_folder_path)
        print(f'Saving image processor to {pytorch_dump_folder_path}')
        processor.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        model.push_to_hub(f'caidas/{model_name}')
        processor.push_to_hub(f'caidas/{model_name}')