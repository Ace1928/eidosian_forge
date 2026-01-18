import argparse
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, RobertaPreLayerNormConfig, RobertaPreLayerNormForMaskedLM
from transformers.utils import logging
def convert_roberta_prelayernorm_checkpoint_to_pytorch(checkpoint_repo: str, pytorch_dump_folder_path: str):
    """
    Copy/paste/tweak roberta_prelayernorm's weights to our BERT structure.
    """
    config = RobertaPreLayerNormConfig.from_pretrained(checkpoint_repo, architectures=['RobertaPreLayerNormForMaskedLM'])
    original_state_dict = torch.load(hf_hub_download(repo_id=checkpoint_repo, filename='pytorch_model.bin'))
    state_dict = {}
    for tensor_key, tensor_value in original_state_dict.items():
        if tensor_key.startswith('roberta.'):
            tensor_key = 'roberta_prelayernorm.' + tensor_key[len('roberta.'):]
        if tensor_key.endswith('.self.LayerNorm.weight') or tensor_key.endswith('.self.LayerNorm.bias'):
            continue
        state_dict[tensor_key] = tensor_value
    model = RobertaPreLayerNormForMaskedLM.from_pretrained(pretrained_model_name_or_path=None, config=config, state_dict=state_dict)
    model.save_pretrained(pytorch_dump_folder_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_repo)
    tokenizer.save_pretrained(pytorch_dump_folder_path)