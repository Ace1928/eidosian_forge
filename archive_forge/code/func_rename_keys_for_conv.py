import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Iterator, List, Set, Tuple
import requests
import torch
import torchvision.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from PIL import Image
from torch import Tensor, nn
from transformers.models.maskformer.feature_extraction_maskformer import MaskFormerImageProcessor
from transformers.models.maskformer.modeling_maskformer import (
from transformers.utils import logging
def rename_keys_for_conv(detectron_conv: str, mine_conv: str):
    return [(f'{detectron_conv}.weight', f'{mine_conv}.0.weight'), (f'{detectron_conv}.norm.weight', f'{mine_conv}.1.weight'), (f'{detectron_conv}.norm.bias', f'{mine_conv}.1.bias')]