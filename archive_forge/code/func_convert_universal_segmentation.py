import json
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
from detectron2.projects.deeplab import add_deeplab_config
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import Tensor, nn
from transformers import (
from transformers.models.mask2former.modeling_mask2former import (
from transformers.utils import logging
def convert_universal_segmentation(self, mask2former: Mask2FormerForUniversalSegmentation) -> Mask2FormerForUniversalSegmentation:
    dst_state_dict = TrackedStateDict(mask2former.state_dict())
    src_state_dict = self.original_model.state_dict()
    self.replace_universal_segmentation_module(dst_state_dict, src_state_dict)
    state_dict = {key: dst_state_dict[key] for key in dst_state_dict.to_track.keys()}
    mask2former.load_state_dict(state_dict)
    return mask2former