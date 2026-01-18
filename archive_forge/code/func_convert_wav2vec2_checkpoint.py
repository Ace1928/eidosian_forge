import argparse
import json
import os
import fairseq
import torch
from fairseq.data import Dictionary
from transformers import (
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForSequenceClassification
@torch.no_grad()
def convert_wav2vec2_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None, dict_path=None, is_finetuned=True, is_seq_class=False):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    if config_path is not None:
        config = Wav2Vec2Config.from_pretrained(config_path)
    else:
        config = Wav2Vec2Config()
    if is_seq_class:
        id2label = read_txt_into_dict(dict_path)
        config.id2label = id2label
        hf_wav2vec = Wav2Vec2ForSequenceClassification(config)
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0, do_normalize=True, return_attention_mask=True)
        feature_extractor.save_pretrained(pytorch_dump_folder_path)
    elif is_finetuned:
        if dict_path:
            target_dict = Dictionary.load(dict_path)
            config.bos_token_id = target_dict.pad_index
            config.pad_token_id = target_dict.bos_index
            config.eos_token_id = target_dict.eos_index
            config.vocab_size = len(target_dict.symbols)
            vocab_path = os.path.join(pytorch_dump_folder_path, 'vocab.json')
            if not os.path.isdir(pytorch_dump_folder_path):
                logger.error('--pytorch_dump_folder_path ({}) should be a directory'.format(pytorch_dump_folder_path))
                return
            os.makedirs(pytorch_dump_folder_path, exist_ok=True)
            vocab_dict = target_dict.indices
            vocab_dict['<pad>'] = 0
            vocab_dict['<s>'] = 1
            with open(vocab_path, 'w', encoding='utf-8') as vocab_handle:
                json.dump(vocab_dict, vocab_handle)
            tokenizer = Wav2Vec2CTCTokenizer(vocab_path, unk_token=target_dict.unk_word, pad_token=target_dict.pad_word, bos_token=target_dict.bos_word, eos_token=target_dict.eos_word, word_delimiter_token='|', do_lower_case=False)
            return_attention_mask = True if config.feat_extract_norm == 'layer' else False
            feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0, do_normalize=True, return_attention_mask=return_attention_mask)
            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            processor.save_pretrained(pytorch_dump_folder_path)
        hf_wav2vec = Wav2Vec2ForCTC(config)
    else:
        hf_wav2vec = Wav2Vec2ForPreTraining(config)
    if is_finetuned or is_seq_class:
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path], arg_overrides={'data': '/'.join(dict_path.split('/')[:-1])})
    else:
        task_arg = argparse.Namespace(task='audio_pretraining')
        task = fairseq.tasks.setup_task(task_arg)
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path], task=task)
    model = model[0].eval()
    recursively_load_weights(model, hf_wav2vec, not is_finetuned)
    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)