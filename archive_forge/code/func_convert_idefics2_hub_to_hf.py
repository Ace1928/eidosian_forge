import argparse
import copy
import torch
from accelerate import init_empty_weights
from transformers import (
def convert_idefics2_hub_to_hf(original_model_id, output_hub_path, push_to_hub):
    original_model = AutoModelForCausalLM.from_pretrained(original_model_id, trust_remote_code=True)
    image_seq_len = original_model.config.perceiver_config.resampler_n_latents
    image_processor = Idefics2ImageProcessor()
    tokenizer = AutoTokenizer.from_pretrained(original_model_id)
    processor = Idefics2Processor(image_processor=image_processor, tokenizer=tokenizer, image_seq_len=image_seq_len)
    state_dict = original_model.state_dict()
    state_dict = convert_state_dict_to_hf(state_dict)
    state_dict = merge_weights(state_dict)
    config = get_config(original_model_id)
    with init_empty_weights():
        model = Idefics2ForConditionalGeneration(config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.save_pretrained(output_hub_path)
    processor.save_pretrained(output_hub_path)
    if push_to_hub:
        model.push_to_hub(output_hub_path, private=True)
        processor.push_to_hub(output_hub_path, private=True)