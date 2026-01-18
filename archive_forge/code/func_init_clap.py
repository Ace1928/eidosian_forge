import argparse
import re
from laion_clap import CLAP_Module
from transformers import AutoFeatureExtractor, ClapConfig, ClapModel
def init_clap(checkpoint_path, model_type, enable_fusion=False):
    model = CLAP_Module(amodel=model_type, enable_fusion=enable_fusion)
    model.load_ckpt(checkpoint_path)
    return model