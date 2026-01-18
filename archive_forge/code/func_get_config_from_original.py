import argparse
import re
from laion_clap import CLAP_Module
from transformers import AutoFeatureExtractor, ClapConfig, ClapModel
def get_config_from_original(clap_model):
    audio_config = {'patch_embeds_hidden_size': clap_model.model.audio_branch.embed_dim, 'depths': clap_model.model.audio_branch.depths, 'hidden_size': clap_model.model.audio_projection[0].in_features}
    text_config = {'hidden_size': clap_model.model.text_branch.pooler.dense.in_features}
    return ClapConfig(audio_config=audio_config, text_config=text_config)