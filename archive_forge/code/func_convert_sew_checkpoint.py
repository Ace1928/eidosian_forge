import argparse
import json
import os
import fairseq
import torch
from fairseq.data import Dictionary
from sew_asapp import tasks  # noqa: F401
from transformers import (
@torch.no_grad()
def convert_sew_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None, dict_path=None, is_finetuned=True):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    if is_finetuned:
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path], arg_overrides={'data': '/'.join(dict_path.split('/')[:-1])})
    else:
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
    if config_path is not None:
        config = SEWDConfig.from_pretrained(config_path)
    else:
        config = convert_config(model[0], is_finetuned)
    model = model[0].eval()
    return_attention_mask = True if config.feat_extract_norm == 'layer' else False
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0, do_normalize=True, return_attention_mask=return_attention_mask)
    if is_finetuned:
        if dict_path:
            target_dict = Dictionary.load(dict_path)
            target_dict.indices[target_dict.bos_word] = target_dict.pad_index
            target_dict.indices[target_dict.pad_word] = target_dict.bos_index
            config.bos_token_id = target_dict.pad_index
            config.pad_token_id = target_dict.bos_index
            config.eos_token_id = target_dict.eos_index
            config.vocab_size = len(target_dict.symbols)
            vocab_path = os.path.join(pytorch_dump_folder_path, 'vocab.json')
            if not os.path.isdir(pytorch_dump_folder_path):
                logger.error('--pytorch_dump_folder_path ({}) should be a directory'.format(pytorch_dump_folder_path))
                return
            os.makedirs(pytorch_dump_folder_path, exist_ok=True)
            with open(vocab_path, 'w', encoding='utf-8') as vocab_handle:
                json.dump(target_dict.indices, vocab_handle)
            tokenizer = Wav2Vec2CTCTokenizer(vocab_path, unk_token=target_dict.unk_word, pad_token=target_dict.pad_word, bos_token=target_dict.bos_word, eos_token=target_dict.eos_word, word_delimiter_token='|', do_lower_case=False)
            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            processor.save_pretrained(pytorch_dump_folder_path)
        hf_model = SEWDForCTC(config)
    else:
        hf_model = SEWDModel(config)
        feature_extractor.save_pretrained(pytorch_dump_folder_path)
    recursively_load_weights(model, hf_model, is_finetuned)
    hf_model.save_pretrained(pytorch_dump_folder_path)