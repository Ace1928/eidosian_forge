import argparse
import torch
from transformers import (
from transformers.tokenization_utils import AddedToken
@torch.no_grad()
def convert_speecht5_checkpoint(task, checkpoint_path, pytorch_dump_folder_path, config_path=None, vocab_path=None, repo_id=None):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    if config_path is not None:
        config = SpeechT5Config.from_pretrained(config_path)
    else:
        config = SpeechT5Config()
    if task == 's2t':
        config.max_length = config.max_text_positions
        model = SpeechT5ForSpeechToText(config)
    elif task == 't2s':
        config.max_speech_positions = 1876
        config.max_text_positions = 600
        config.max_length = config.max_speech_positions
        model = SpeechT5ForTextToSpeech(config)
    elif task == 's2s':
        config.max_speech_positions = 1876
        config.max_length = config.max_speech_positions
        model = SpeechT5ForSpeechToSpeech(config)
    else:
        raise ValueError(f'Unknown task name: {task}')
    if vocab_path:
        tokenizer = SpeechT5Tokenizer(vocab_path, model_max_length=config.max_text_positions)
        mask_token = AddedToken('<mask>', lstrip=True, rstrip=False)
        tokenizer.mask_token = mask_token
        tokenizer.add_special_tokens({'mask_token': mask_token})
        tokenizer.add_tokens(['<ctc_blank>'])
    feature_extractor = SpeechT5FeatureExtractor()
    processor = SpeechT5Processor(tokenizer=tokenizer, feature_extractor=feature_extractor)
    processor.save_pretrained(pytorch_dump_folder_path)
    fairseq_checkpoint = torch.load(checkpoint_path)
    recursively_load_weights(fairseq_checkpoint['model'], model, task)
    model.save_pretrained(pytorch_dump_folder_path)
    if repo_id:
        print('Pushing to the hub...')
        processor.push_to_hub(repo_id)
        model.push_to_hub(repo_id)