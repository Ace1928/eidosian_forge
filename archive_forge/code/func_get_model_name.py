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
def get_model_name(checkpoint_file: Path):
    model_name_raw: str = checkpoint_file.parents[0].stem
    segmentation_task_name: str = checkpoint_file.parents[1].stem
    if segmentation_task_name not in ['instance-segmentation', 'panoptic-segmentation', 'semantic-segmentation']:
        raise ValueError(f'{segmentation_task_name} must be wrong since acceptable values are: instance-segmentation, panoptic-segmentation, semantic-segmentation.')
    dataset_name: str = checkpoint_file.parents[2].stem
    if dataset_name not in ['coco', 'ade', 'cityscapes', 'mapillary-vistas']:
        raise ValueError(f"{dataset_name} must be wrong since we didn't find 'coco' or 'ade' or 'cityscapes' or 'mapillary-vistas' in it ")
    backbone = 'swin'
    backbone_types = ['tiny', 'small', 'base_IN21k', 'base', 'large']
    backbone_type = list(filter(lambda x: x in model_name_raw, backbone_types))[0].replace('_', '-')
    model_name = f'mask2former-{backbone}-{backbone_type}-{dataset_name}-{segmentation_task_name.split('-')[0]}'
    return model_name