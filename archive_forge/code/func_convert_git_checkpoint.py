import argparse
from pathlib import Path
import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import (
from transformers.utils import logging
@torch.no_grad()
def convert_git_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our GIT structure.
    """
    model_name_to_url = {'git-base': 'https://publicgit.blob.core.windows.net/data/output/GIT_BASE/snapshot/model.pt', 'git-base-coco': 'https://publicgit.blob.core.windows.net/data/output/GIT_BASE_COCO/snapshot/model.pt', 'git-base-textcaps': 'https://publicgit.blob.core.windows.net/data/output/GIT_BASE_TEXTCAPS/snapshot/model.pt', 'git-base-vqav2': 'https://publicgit.blob.core.windows.net/data/output/GIT_BASE_VQAv2/snapshot/model.pt', 'git-base-textvqa': 'https://publicgit.blob.core.windows.net/data/output/GIT_BASE_TEXTVQA/snapshot/model.pt', 'git-base-vatex': 'https://publicgit.blob.core.windows.net/data/output/GIT_BASE_VATEX/snapshot/model.pt', 'git-base-msrvtt-qa': 'https://publicgit.blob.core.windows.net/data/output/GIT_BASE_MSRVTT_QA/snapshot/model.pt', 'git-large': 'https://publicgit.blob.core.windows.net/data/output/GIT_LARGE/snapshot/model.pt', 'git-large-coco': 'https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_COCO/snapshot/model.pt', 'git-large-textcaps': 'https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_TEXTCAPS/snapshot/model.pt', 'git-large-vqav2': 'https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_VQAv2/snapshot/model.pt', 'git-large-textvqa': 'https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_TEXTVQA/snapshot/model.pt', 'git-large-vatex': 'https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_VATEX/snapshot/model.pt', 'git-large-msrvtt-qa': 'https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_MSRVTT_QA/snapshot/model.pt', 'git-large-r': 'https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_R/snapshot/model.pt', 'git-large-r-coco': 'https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_R_COCO/snapshot/model.pt', 'git-large-r-textcaps': 'https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_R_TEXTCAPS/snapshot/model.pt'}
    model_name_to_path = {'git-large': '/Users/nielsrogge/Documents/GIT/git_large_model.pt', 'git-large-coco': '/Users/nielsrogge/Documents/GIT/git_large_coco_model.pt', 'git-large-textcaps': '/Users/nielsrogge/Documents/GIT/git_large_textcaps_model.pt', 'git-large-vqav2': '/Users/nielsrogge/Documents/GIT/git_large_vqav2_model.pt', 'git-large-textvqa': '/Users/nielsrogge/Documents/GIT/git_large_textvqa_model.pt'}
    config, image_size, is_video = get_git_config(model_name)
    if 'large' in model_name and (not is_video) and ('large-r' not in model_name):
        checkpoint_path = model_name_to_path[model_name]
        state_dict = torch.load(checkpoint_path, map_location='cpu')['model']
    else:
        checkpoint_url = model_name_to_url[model_name]
        state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu', file_name=model_name)['model']
    prefix = 'module.' if model_name == 'git-base' else ''
    rename_keys = create_rename_keys(config, prefix=prefix)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config, prefix=prefix)
    model = GitForCausalLM(config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    model.eval()
    print('Missing keys:', missing_keys)
    print('Unexpected keys:', unexpected_keys)
    assert missing_keys == ['git.embeddings.position_ids', 'git.image_encoder.vision_model.embeddings.position_ids']
    assert unexpected_keys == ['git.image_encoder.visual_projection.weight']
    image_processor = VideoMAEImageProcessor(size={'shortest_edge': image_size}, crop_size={'height': image_size, 'width': image_size}) if is_video else CLIPImageProcessor(size={'shortest_edge': image_size}, crop_size={'height': image_size, 'width': image_size})
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased', model_input_names=['input_ids', 'attention_mask'])
    processor = GitProcessor(tokenizer=tokenizer, image_processor=image_processor)
    if is_video:
        video = prepare_video()
        pixel_values = processor(images=list(video), return_tensors='pt').pixel_values
    else:
        image = prepare_img(model_name)
        image_transforms = Compose([Resize(image_size, interpolation=Image.BICUBIC), CenterCrop(image_size), ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        original_pixel_values = image_transforms(image).unsqueeze(0)
        pixel_values = processor(images=image, return_tensors='pt').pixel_values
        assert torch.allclose(pixel_values, original_pixel_values)
    input_ids = torch.tensor([[101]])
    outputs = model(input_ids, pixel_values=pixel_values)
    logits = outputs.logits
    print('Logits:', logits[0, -1, :3])
    if model_name == 'git-base':
        expected_slice_logits = torch.tensor([-1.2832, -1.2835, -1.284])
    elif model_name == 'git-base-coco':
        expected_slice_logits = torch.tensor([-0.9925, -0.993, -0.9935])
    elif model_name == 'git-base-textcaps':
        expected_slice_logits = torch.tensor([-1.298, -1.2983, -1.2985])
    elif model_name == 'git-base-vqav2':
        expected_slice_logits = torch.tensor([-0.857, -0.8568, -0.8561])
    elif model_name == 'git-base-textvqa':
        expected_slice_logits = torch.tensor([-1.4085, -1.4083, -1.4082])
    elif model_name == 'git-base-vatex':
        expected_slice_logits = torch.tensor([-1.3451, -1.3447, -1.3447])
    elif model_name == 'git-base-msrvtt-qa':
        expected_slice_logits = torch.tensor([-0.8554, -0.855, -0.854])
    elif model_name == 'git-large':
        expected_slice_logits = torch.tensor([-1.1708, -1.1707, -1.1705])
    elif model_name == 'git-large-coco':
        expected_slice_logits = torch.tensor([-1.0425, -1.0423, -1.0422])
    elif model_name == 'git-large-textcaps':
        expected_slice_logits = torch.tensor([-1.2705, -1.2708, -1.2706])
    elif model_name == 'git-large-vqav2':
        expected_slice_logits = torch.tensor([-0.7042, -0.7043, -0.7043])
    elif model_name == 'git-large-textvqa':
        expected_slice_logits = torch.tensor([-0.859, -0.8592, -0.859])
    elif model_name == 'git-large-vatex':
        expected_slice_logits = torch.tensor([-1.0113, -1.0114, -1.0113])
    elif model_name == 'git-large-msrvtt-qa':
        expected_slice_logits = torch.tensor([0.013, 0.0134, 0.0131])
    elif model_name == 'git-large-r':
        expected_slice_logits = torch.tensor([-1.1283, -1.1285, -1.1286])
    elif model_name == 'git-large-r-coco':
        expected_slice_logits = torch.tensor([-0.9641, -0.9641, -0.9641])
    elif model_name == 'git-large-r-textcaps':
        expected_slice_logits = torch.tensor([-1.1121, -1.112, -1.1124])
    assert torch.allclose(logits[0, -1, :3], expected_slice_logits, atol=0.0001)
    print('Looks ok!')
    prompt = ''
    if 'textvqa' in model_name:
        prompt = 'what does the front of the bus say at the top?'
    elif 'msrvtt-qa' in model_name:
        prompt = 'what does the woman eat?'
    elif 'vqa' in model_name:
        prompt = 'what are the cats doing?'
    input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    input_ids = [processor.tokenizer.cls_token_id] + input_ids
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    print('Generating caption...')
    generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
    print('Generated caption:', processor.batch_decode(generated_ids, skip_special_tokens=True))
    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f'Saving model and processor of {model_name} to {pytorch_dump_folder_path}')
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        print(f'Pushing model and processor of {model_name} to the hub...')
        model.push_to_hub(f'microsoft/{model_name}')
        processor.push_to_hub(f'microsoft/{model_name}')