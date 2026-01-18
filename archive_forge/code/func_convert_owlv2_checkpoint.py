import argparse
import collections
import os
import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax.training import checkpoints
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (
from transformers.utils import logging
@torch.no_grad()
def convert_owlv2_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub, verify_logits):
    """
    Copy/paste/tweak model's weights to our OWL-ViT structure.
    """
    config = get_owlv2_config(model_name)
    variables = checkpoints.restore_checkpoint(checkpoint_path, target=None)
    variables = variables['params'] if 'v2' in model_name else variables['optimizer']['target']
    flax_params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, variables)
    state_dict = flatten_nested_dict(flax_params)
    rename_keys = create_rename_keys(config, model_name)
    for src, dest in rename_keys:
        rename_and_reshape_key(state_dict, src, dest, config)
    model = Owlv2ForObjectDetection(config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert missing_keys == ['owlv2.visual_projection.weight']
    assert unexpected_keys == []
    model.eval()
    size = {'height': config.vision_config.image_size, 'width': config.vision_config.image_size}
    image_processor = Owlv2ImageProcessor(size=size)
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32', pad_token='!', model_max_length=16)
    processor = Owlv2Processor(image_processor=image_processor, tokenizer=tokenizer)
    filepath = hf_hub_download(repo_id='nielsr/test-image', filename='owlvit_pixel_values_960.pt', repo_type='dataset')
    original_pixel_values = torch.load(filepath).permute(0, 3, 1, 2)
    filepath = hf_hub_download(repo_id='nielsr/test-image', filename='owlv2_input_ids.pt', repo_type='dataset')
    original_input_ids = torch.load(filepath).squeeze()
    filepath = hf_hub_download(repo_id='adirik/OWL-ViT', repo_type='space', filename='assets/astronaut.png')
    image = Image.open(filepath)
    texts = [['face', 'rocket', 'nasa badge', 'star-spangled banner']]
    inputs = processor(text=texts, images=image, return_tensors='pt')
    if 'large' not in model_name:
        assert torch.allclose(inputs.pixel_values, original_pixel_values.float(), atol=1e-06)
    assert torch.allclose(inputs.input_ids[:4, :], original_input_ids[:4, :], atol=1e-06)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_boxes = outputs.pred_boxes
        objectness_logits = outputs.objectness_logits
    if verify_logits:
        if model_name == 'owlv2-base-patch16':
            expected_logits = torch.tensor([[-10.0043, -9.0226, -8.0433], [-12.4569, -14.038, -12.6153], [-21.0731, -22.2705, -21.885]])
            expected_boxes = torch.tensor([[0.0136, 0.0223, 0.0269], [0.0406, 0.0327, 0.0797], [0.0638, 0.1539, 0.1255]])
            expected_objectness_logits = torch.tensor([[-5.6589, -7.7702, -16.3965]])
        elif model_name == 'owlv2-base-patch16-finetuned':
            expected_logits = torch.tensor([[-9.2391, -9.2313, -8.0295], [-14.5498, -16.845, -14.7166], [-15.1278, -17.306, -15.7169]])
            expected_boxes = torch.tensor([[0.0103, 0.0094, 0.0207], [0.0483, 0.0729, 0.1013], [0.0629, 0.1396, 0.1313]])
            expected_objectness_logits = torch.tensor([[-6.5234, -13.3788, -14.6627]])
        elif model_name == 'owlv2-base-patch16-ensemble':
            expected_logits = torch.tensor([[-8.6353, -9.5409, -6.6154], [-7.9442, -9.6151, -6.7117], [-12.4593, -15.3332, -12.1048]])
            expected_boxes = torch.tensor([[0.0126, 0.009, 0.0238], [0.0387, 0.0227, 0.0754], [0.0582, 0.1058, 0.1139]])
            expected_objectness_logits = torch.tensor([[-6.0628, -5.9507, -10.4486]])
        elif model_name == 'owlv2-large-patch14':
            expected_logits = torch.tensor([[-12.6662, -11.8384, -12.188], [-16.0599, -16.5835, -16.9364], [-21.4957, -26.7038, -25.1313]])
            expected_boxes = torch.tensor([[0.0136, 0.0161, 0.0256], [0.0126, 0.0135, 0.0202], [0.0498, 0.0948, 0.0915]])
            expected_objectness_logits = torch.tensor([[-6.7196, -9.459, -13.9472]])
        elif model_name == 'owlv2-large-patch14-finetuned':
            expected_logits = torch.tensor([[-9.5413, -9.713, -7.9762], [-9.5731, -9.7277, -8.2252], [-15.4434, -19.3084, -16.549]])
            expected_boxes = torch.tensor([[0.0089, 0.008, 0.0175], [0.0112, 0.0098, 0.0179], [0.0375, 0.0821, 0.0528]])
            expected_objectness_logits = torch.tensor([[-6.2655, -6.5845, -11.3105]])
        elif model_name == 'owlv2-large-patch14-ensemble':
            expected_logits = torch.tensor([[-12.2037, -12.207, -11.5371], [-13.4875, -13.8235, -13.1586], [-18.2007, -22.9834, -20.6816]])
            expected_boxes = torch.tensor([[0.0126, 0.0127, 0.0222], [0.0107, 0.0113, 0.0164], [0.0482, 0.1162, 0.0885]])
            expected_objectness_logits = torch.tensor([[-7.7572, -8.3637, -13.0334]])
        print('Objectness logits:', objectness_logits[:3, :3])
        print('Logits:', logits[0, :3, :3])
        print('Pred boxes:', pred_boxes[0, :3, :3])
        assert torch.allclose(logits[0, :3, :3], expected_logits, atol=0.001)
        assert torch.allclose(pred_boxes[0, :3, :3], expected_boxes, atol=0.001)
        assert torch.allclose(objectness_logits[:3, :3], expected_objectness_logits, atol=0.001)
        print('Looks ok!')
    else:
        print('Model converted without verifying logits')
    if pytorch_dump_folder_path is not None:
        print('Saving model and processor locally...')
        if not os.path.isdir(pytorch_dump_folder_path):
            os.mkdir(pytorch_dump_folder_path)
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        print(f'Pushing {model_name} to the hub...')
        model.push_to_hub(f'google/{model_name}')
        processor.push_to_hub(f'google/{model_name}')