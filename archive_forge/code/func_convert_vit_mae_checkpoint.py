import argparse
import requests
import torch
from PIL import Image
from transformers import ViTMAEConfig, ViTMAEForPreTraining, ViTMAEImageProcessor
def convert_vit_mae_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    config = ViTMAEConfig()
    if 'large' in checkpoint_url:
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
    elif 'huge' in checkpoint_url:
        config.patch_size = 14
        config.hidden_size = 1280
        config.intermediate_size = 5120
        config.num_hidden_layers = 32
        config.num_attention_heads = 16
    model = ViTMAEForPreTraining(config)
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu')['model']
    image_processor = ViTMAEImageProcessor(size=config.image_size)
    new_state_dict = convert_state_dict(state_dict, config)
    model.load_state_dict(new_state_dict)
    model.eval()
    url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    image_processor = ViTMAEImageProcessor(size=config.image_size)
    inputs = image_processor(images=image, return_tensors='pt')
    torch.manual_seed(2)
    outputs = model(**inputs)
    logits = outputs.logits
    if 'large' in checkpoint_url:
        expected_slice = torch.tensor([[-0.7309, -0.7128, -1.0169], [-1.0161, -0.9058, -1.1878], [-1.0478, -0.9411, -1.1911]])
    elif 'huge' in checkpoint_url:
        expected_slice = torch.tensor([[-1.1599, -0.9199, -1.2221], [-1.1952, -0.9269, -1.2307], [-1.2143, -0.9337, -1.2262]])
    else:
        expected_slice = torch.tensor([[-0.9192, -0.8481, -1.1259], [-1.1349, -1.0034, -1.2599], [-1.1757, -1.0429, -1.2726]])
    assert torch.allclose(logits[0, :3, :3], expected_slice, atol=0.0001)
    print(f'Saving model to {pytorch_dump_folder_path}')
    model.save_pretrained(pytorch_dump_folder_path)
    print(f'Saving image processor to {pytorch_dump_folder_path}')
    image_processor.save_pretrained(pytorch_dump_folder_path)