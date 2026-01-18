import argparse
import re
from laion_clap import CLAP_Module
from transformers import AutoFeatureExtractor, ClapConfig, ClapModel
def convert_clap_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path, model_type, enable_fusion=False):
    clap_model = init_clap(checkpoint_path, model_type, enable_fusion=enable_fusion)
    clap_model.eval()
    state_dict = clap_model.model.state_dict()
    state_dict = rename_state_dict(state_dict)
    transformers_config = get_config_from_original(clap_model)
    transformers_config.audio_config.enable_fusion = enable_fusion
    model = ClapModel(transformers_config)
    model.load_state_dict(state_dict, strict=False)
    model.save_pretrained(pytorch_dump_folder_path)
    transformers_config.save_pretrained(pytorch_dump_folder_path)