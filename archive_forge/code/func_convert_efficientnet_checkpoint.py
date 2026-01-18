import argparse
import json
import os
import numpy as np
import PIL
import requests
import tensorflow.keras.applications.efficientnet as efficientnet
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from tensorflow.keras.preprocessing import image
from transformers import (
from transformers.utils import logging
@torch.no_grad()
def convert_efficientnet_checkpoint(model_name, pytorch_dump_folder_path, save_model, push_to_hub):
    """
    Copy/paste/tweak model's weights to our EfficientNet structure.
    """
    original_model = model_classes[model_name](include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation='softmax')
    tf_params = original_model.trainable_variables
    tf_non_train_params = original_model.non_trainable_variables
    tf_params = {param.name: param.numpy() for param in tf_params}
    for param in tf_non_train_params:
        tf_params[param.name] = param.numpy()
    tf_param_names = list(tf_params.keys())
    config = get_efficientnet_config(model_name)
    hf_model = EfficientNetForImageClassification(config).eval()
    hf_params = hf_model.state_dict()
    print('Converting parameters...')
    key_mapping = rename_keys(tf_param_names)
    replace_params(hf_params, tf_params, key_mapping)
    preprocessor = convert_image_processor(model_name)
    inputs = preprocessor(images=prepare_img(), return_tensors='pt')
    hf_model.eval()
    with torch.no_grad():
        outputs = hf_model(**inputs)
    hf_logits = outputs.logits.detach().numpy()
    original_model.trainable = False
    image_size = CONFIG_MAP[model_name]['image_size']
    img = prepare_img().resize((image_size, image_size), resample=PIL.Image.NEAREST)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    original_logits = original_model.predict(x)
    assert np.allclose(original_logits, hf_logits, atol=0.001), 'The predicted logits are not the same.'
    print('Model outputs match!')
    if save_model:
        if not os.path.isdir(pytorch_dump_folder_path):
            os.mkdir(pytorch_dump_folder_path)
        hf_model.save_pretrained(pytorch_dump_folder_path)
        preprocessor.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        print(f'Pushing converted {model_name} to the hub...')
        model_name = f'efficientnet-{model_name}'
        preprocessor.push_to_hub(model_name)
        hf_model.push_to_hub(model_name)