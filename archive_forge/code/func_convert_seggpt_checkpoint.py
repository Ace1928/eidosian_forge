import argparse
import requests
import torch
from PIL import Image
from transformers import SegGptConfig, SegGptForImageSegmentation, SegGptImageProcessor
from transformers.utils import logging
@torch.no_grad()
def convert_seggpt_checkpoint(args):
    model_name = args.model_name
    pytorch_dump_folder_path = args.pytorch_dump_folder_path
    verify_logits = args.verify_logits
    push_to_hub = args.push_to_hub
    config = SegGptConfig()
    checkpoint_url = 'https://huggingface.co/BAAI/SegGpt/blob/main/seggpt_vit_large.pth'
    original_state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu')['model']
    new_state_dict = original_state_dict.copy()
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(new_state_dict, src, dest)
    model = SegGptForImageSegmentation(config)
    model.eval()
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print('Missing keys:', missing_keys)
    print('Unexpected keys:', unexpected_keys)
    input_img, prompt_img, prompt_mask = prepare_input()
    image_processor = SegGptImageProcessor()
    inputs = image_processor(images=input_img, prompt_images=prompt_img, prompt_masks=prompt_mask, return_tensors='pt')
    expected_prompt_pixel_values = torch.tensor([[[-0.6965, -0.6965, -0.6965], [-0.6965, -0.6965, -0.6965], [-0.6965, -0.6965, -0.6965]], [[1.6583, 1.6583, 1.6583], [1.6583, 1.6583, 1.6583], [1.6583, 1.6583, 1.6583]], [[2.3088, 2.3088, 2.3088], [2.3088, 2.3088, 2.3088], [2.3088, 2.3088, 2.3088]]])
    expected_pixel_values = torch.tensor([[[1.6324, 1.6153, 1.581], [1.6153, 1.5982, 1.581], [1.581, 1.5639, 1.5639]], [[1.2731, 1.2556, 1.2206], [1.2556, 1.2381, 1.2031], [1.2206, 1.2031, 1.1681]], [[1.6465, 1.6465, 1.6465], [1.6465, 1.6465, 1.6465], [1.6291, 1.6291, 1.6291]]])
    expected_prompt_masks = torch.tensor([[[-2.1179, -2.1179, -2.1179], [-2.1179, -2.1179, -2.1179], [-2.1179, -2.1179, -2.1179]], [[-2.0357, -2.0357, -2.0357], [-2.0357, -2.0357, -2.0357], [-2.0357, -2.0357, -2.0357]], [[-1.8044, -1.8044, -1.8044], [-1.8044, -1.8044, -1.8044], [-1.8044, -1.8044, -1.8044]]])
    assert torch.allclose(inputs.pixel_values[0, :, :3, :3], expected_pixel_values, atol=0.0001)
    assert torch.allclose(inputs.prompt_pixel_values[0, :, :3, :3], expected_prompt_pixel_values, atol=0.0001)
    assert torch.allclose(inputs.prompt_masks[0, :, :3, :3], expected_prompt_masks, atol=0.0001)
    torch.manual_seed(2)
    outputs = model(**inputs)
    print(outputs)
    if verify_logits:
        expected_output = torch.tensor([[[-2.1208, -2.119, -2.1198], [-2.1237, -2.1228, -2.1227], [-2.1232, -2.1226, -2.1228]], [[-2.0405, -2.0396, -2.0403], [-2.0434, -2.0434, -2.0433], [-2.0428, -2.0432, -2.0434]], [[-1.8102, -1.8088, -1.8099], [-1.8131, -1.8126, -1.8129], [-1.813, -1.8128, -1.8131]]])
        assert torch.allclose(outputs.pred_masks[0, :, :3, :3], expected_output, atol=0.0001)
        print('Looks good!')
    else:
        print('Converted without verifying logits')
    if pytorch_dump_folder_path is not None:
        print(f'Saving model and processor for {model_name} to {pytorch_dump_folder_path}')
        model.save_pretrained(pytorch_dump_folder_path)
        image_processor.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        print(f'Pushing model and processor for {model_name} to hub')
        model.push_to_hub(f'EduardoPacheco/{model_name}')
        image_processor.push_to_hub(f'EduardoPacheco/{model_name}')