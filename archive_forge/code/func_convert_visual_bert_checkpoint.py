import argparse
from collections import OrderedDict
from pathlib import Path
import torch
from transformers import (
from transformers.utils import logging
@torch.no_grad()
def convert_visual_bert_checkpoint(checkpoint_path, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our VisualBERT structure.
    """
    assert checkpoint_path.split('/')[-1] in ACCEPTABLE_CHECKPOINTS, f'The checkpoint provided must be in {ACCEPTABLE_CHECKPOINTS}.'
    if 'pre' in checkpoint_path:
        model_type = 'pretraining'
        if 'vcr' in checkpoint_path:
            config_params = {'visual_embedding_dim': 512}
        elif 'vqa_advanced' in checkpoint_path:
            config_params = {'visual_embedding_dim': 2048}
        elif 'vqa' in checkpoint_path:
            config_params = {'visual_embedding_dim': 2048}
        elif 'nlvr' in checkpoint_path:
            config_params = {'visual_embedding_dim': 1024}
        else:
            raise NotImplementedError(f'No implementation found for `{checkpoint_path}`.')
    elif 'vcr' in checkpoint_path:
        config_params = {'visual_embedding_dim': 512}
        model_type = 'multichoice'
    elif 'vqa_advanced' in checkpoint_path:
        config_params = {'visual_embedding_dim': 2048}
        model_type = 'vqa_advanced'
    elif 'vqa' in checkpoint_path:
        config_params = {'visual_embedding_dim': 2048, 'num_labels': 3129}
        model_type = 'vqa'
    elif 'nlvr' in checkpoint_path:
        config_params = {'visual_embedding_dim': 1024, 'num_labels': 2}
        model_type = 'nlvr'
    config = VisualBertConfig(**config_params)
    state_dict = load_state_dict(checkpoint_path)
    new_state_dict = get_new_dict(state_dict, config)
    if model_type == 'pretraining':
        model = VisualBertForPreTraining(config)
    elif model_type == 'vqa':
        model = VisualBertForQuestionAnswering(config)
    elif model_type == 'nlvr':
        model = VisualBertForVisualReasoning(config)
    elif model_type == 'multichoice':
        model = VisualBertForMultipleChoice(config)
    model.load_state_dict(new_state_dict)
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)