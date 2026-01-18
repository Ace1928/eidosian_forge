import argparse
import torch
from s3prl.hub import distilhubert
from transformers import HubertConfig, HubertModel, Wav2Vec2FeatureExtractor, logging
@torch.no_grad()
def convert_hubert_checkpoint(pytorch_dump_folder_path, config_path=None):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    model = distilhubert().model.model
    if config_path is not None:
        config = HubertConfig.from_pretrained(config_path)
    else:
        config = convert_config(model)
    model = model.eval()
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0, do_normalize=False, return_attention_mask=False)
    hf_model = HubertModel(config)
    recursively_load_weights(model, hf_model)
    feature_extractor.save_pretrained(pytorch_dump_folder_path)
    hf_model.save_pretrained(pytorch_dump_folder_path)