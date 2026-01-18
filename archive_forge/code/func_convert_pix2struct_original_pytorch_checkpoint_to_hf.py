import argparse
import os
import re
import torch
from flax.traverse_util import flatten_dict
from t5x import checkpoints
from transformers import (
def convert_pix2struct_original_pytorch_checkpoint_to_hf(t5x_checkpoint_path, pytorch_dump_folder_path, use_large=False, is_vqa=False):
    flax_params = get_flax_param(t5x_checkpoint_path)
    if not use_large:
        encoder_config = Pix2StructVisionConfig()
        decoder_config = Pix2StructTextConfig()
    else:
        encoder_config = Pix2StructVisionConfig(hidden_size=1536, d_ff=3968, num_attention_heads=24, num_hidden_layers=18)
        decoder_config = Pix2StructTextConfig(hidden_size=1536, d_ff=3968, num_heads=24, num_layers=18)
    config = Pix2StructConfig(vision_config=encoder_config.to_dict(), text_config=decoder_config.to_dict(), is_vqa=is_vqa)
    model = Pix2StructForConditionalGeneration(config)
    torch_params = rename_and_convert_flax_params(flax_params)
    model.load_state_dict(torch_params)
    tok = AutoTokenizer.from_pretrained('ybelkada/test-pix2struct-tokenizer')
    image_processor = Pix2StructImageProcessor()
    processor = Pix2StructProcessor(image_processor=image_processor, tokenizer=tok)
    if use_large:
        processor.image_processor.max_patches = 4096
    processor.image_processor.is_vqa = True
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    processor.save_pretrained(pytorch_dump_folder_path)
    print('Model saved in {}'.format(pytorch_dump_folder_path))