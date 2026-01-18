import argparse
import json
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import SegformerImageProcessor, SwinConfig, UperNetConfig, UperNetForSemanticSegmentation
def reverse_correct_unfold_reduction_order(x):
    out_channel, in_channel = x.shape
    x = x.reshape(out_channel, in_channel // 4, 4)
    x = x[:, :, [0, 2, 1, 3]].transpose(1, 2).reshape(out_channel, in_channel)
    return x