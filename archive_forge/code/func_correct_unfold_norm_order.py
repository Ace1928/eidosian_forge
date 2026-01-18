import argparse
import json
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import SegformerImageProcessor, SwinConfig, UperNetConfig, UperNetForSemanticSegmentation
def correct_unfold_norm_order(x):
    in_channel = x.shape[0]
    x = x.reshape(4, in_channel // 4)
    x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
    return x