import argparse
import json
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import ConvNextConfig, SegformerImageProcessor, UperNetConfig, UperNetForSemanticSegmentation
def convert_upernet_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    model_name_to_url = {'upernet-convnext-tiny': 'https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_tiny_fp16_512x512_160k_ade20k/upernet_convnext_tiny_fp16_512x512_160k_ade20k_20220227_124553-cad485de.pth', 'upernet-convnext-small': 'https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_small_fp16_512x512_160k_ade20k/upernet_convnext_small_fp16_512x512_160k_ade20k_20220227_131208-1b1e394f.pth', 'upernet-convnext-base': 'https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_base_fp16_512x512_160k_ade20k/upernet_convnext_base_fp16_512x512_160k_ade20k_20220227_181227-02a24fc6.pth', 'upernet-convnext-large': 'https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_large_fp16_640x640_160k_ade20k/upernet_convnext_large_fp16_640x640_160k_ade20k_20220226_040532-e57aa54d.pth', 'upernet-convnext-xlarge': 'https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_xlarge_fp16_640x640_160k_ade20k/upernet_convnext_xlarge_fp16_640x640_160k_ade20k_20220226_080344-95fc38c2.pth'}
    checkpoint_url = model_name_to_url[model_name]
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu')['state_dict']
    config = get_upernet_config(model_name)
    model = UperNetForSemanticSegmentation(config)
    model.eval()
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        if 'bn' in key:
            key = key.replace('bn', 'batch_norm')
        state_dict[key] = val
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    model.load_state_dict(state_dict)
    url = 'https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg'
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    processor = SegformerImageProcessor()
    pixel_values = processor(image, return_tensors='pt').pixel_values
    with torch.no_grad():
        outputs = model(pixel_values)
    if model_name == 'upernet-convnext-tiny':
        expected_slice = torch.tensor([[-8.811, -8.811, -8.6521], [-8.811, -8.811, -8.6521], [-8.7746, -8.7746, -8.613]])
    elif model_name == 'upernet-convnext-small':
        expected_slice = torch.tensor([[-8.8236, -8.8236, -8.6771], [-8.8236, -8.8236, -8.6771], [-8.7638, -8.7638, -8.624]])
    elif model_name == 'upernet-convnext-base':
        expected_slice = torch.tensor([[-8.8558, -8.8558, -8.6905], [-8.8558, -8.8558, -8.6905], [-8.7669, -8.7669, -8.6021]])
    elif model_name == 'upernet-convnext-large':
        expected_slice = torch.tensor([[-8.666, -8.666, -8.621], [-8.666, -8.666, -8.621], [-8.631, -8.631, -8.5964]])
    elif model_name == 'upernet-convnext-xlarge':
        expected_slice = torch.tensor([[-8.498, -8.498, -8.3977], [-8.498, -8.498, -8.3977], [-8.4379, -8.4379, -8.3412]])
    print('Logits:', outputs.logits[0, 0, :3, :3])
    assert torch.allclose(outputs.logits[0, 0, :3, :3], expected_slice, atol=0.0001)
    print('Looks ok!')
    if pytorch_dump_folder_path is not None:
        print(f'Saving model {model_name} to {pytorch_dump_folder_path}')
        model.save_pretrained(pytorch_dump_folder_path)
        print(f'Saving processor to {pytorch_dump_folder_path}')
        processor.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        print(f'Pushing model and processor for {model_name} to hub')
        model.push_to_hub(f'openmmlab/{model_name}')
        processor.push_to_hub(f'openmmlab/{model_name}')