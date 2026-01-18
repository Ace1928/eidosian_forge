import argparse
from pathlib import Path
from typing import Dict, OrderedDict, Tuple
import torch
from audiocraft.models import MusicGen
from transformers import (
from transformers.models.musicgen.modeling_musicgen import MusicgenForCausalLM
from transformers.utils import logging
@torch.no_grad()
def convert_musicgen_checkpoint(checkpoint, pytorch_dump_folder=None, repo_id=None, device='cpu', safe_serialization=False):
    fairseq_model = MusicGen.get_pretrained(checkpoint, device=device)
    decoder_config = decoder_config_from_checkpoint(checkpoint)
    decoder_state_dict = fairseq_model.lm.state_dict()
    decoder_state_dict, enc_dec_proj_state_dict = rename_state_dict(decoder_state_dict, hidden_size=decoder_config.hidden_size)
    text_encoder = T5EncoderModel.from_pretrained('google-t5/t5-base')
    audio_encoder = EncodecModel.from_pretrained('facebook/encodec_32khz')
    decoder = MusicgenForCausalLM(decoder_config).eval()
    missing_keys, unexpected_keys = decoder.load_state_dict(decoder_state_dict, strict=False)
    for key in missing_keys.copy():
        if key.startswith(('text_encoder', 'audio_encoder')) or key in EXPECTED_MISSING_KEYS:
            missing_keys.remove(key)
    if len(missing_keys) > 0:
        raise ValueError(f'Missing key(s) in state_dict: {missing_keys}')
    if len(unexpected_keys) > 0:
        raise ValueError(f'Unexpected key(s) in state_dict: {unexpected_keys}')
    model = MusicgenForConditionalGeneration(text_encoder=text_encoder, audio_encoder=audio_encoder, decoder=decoder)
    model.enc_to_dec_proj.load_state_dict(enc_dec_proj_state_dict)
    input_ids = torch.arange(0, 2 * decoder_config.num_codebooks, dtype=torch.long).reshape(2, -1)
    decoder_input_ids = input_ids.reshape(2 * decoder_config.num_codebooks, -1)
    with torch.no_grad():
        logits = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits
    if logits.shape != (2 * decoder_config.num_codebooks, 1, 2048):
        raise ValueError('Incorrect shape for logits')
    tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-base')
    feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/encodec_32khz', padding_side='left', feature_size=decoder_config.audio_channels)
    processor = MusicgenProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model.generation_config.decoder_start_token_id = 2048
    model.generation_config.pad_token_id = 2048
    model.generation_config.max_length = int(30 * audio_encoder.config.frame_rate)
    model.generation_config.do_sample = True
    model.generation_config.guidance_scale = 3.0
    if pytorch_dump_folder is not None:
        Path(pytorch_dump_folder).mkdir(exist_ok=True)
        logger.info(f'Saving model {checkpoint} to {pytorch_dump_folder}')
        model.save_pretrained(pytorch_dump_folder, safe_serialization=safe_serialization)
        processor.save_pretrained(pytorch_dump_folder)
    if repo_id:
        logger.info(f'Pushing model {checkpoint} to {repo_id}')
        model.push_to_hub(repo_id, safe_serialization=safe_serialization)
        processor.push_to_hub(repo_id)