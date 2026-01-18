import argparse
import itertools
import math
from pathlib import Path
import requests
import torch
from PIL import Image
from torchvision import transforms
from transformers import Dinov2Config, DPTConfig, DPTForDepthEstimation, DPTImageProcessor
from transformers.utils import logging
def get_original_pixel_values(image):

    class CenterPadding(object):

        def __init__(self, multiple):
            super().__init__()
            self.multiple = multiple

        def _get_pad(self, size):
            new_size = math.ceil(size / self.multiple) * self.multiple
            pad_size = new_size - size
            pad_size_left = pad_size // 2
            pad_size_right = pad_size - pad_size_left
            return (pad_size_left, pad_size_right)

        def __call__(self, img):
            pads = list(itertools.chain.from_iterable((self._get_pad(m) for m in img.shape[-2:][::-1])))
            output = torch.nn.functional.pad(img, pads)
            return output

        def __repr__(self):
            return self.__class__.__name__ + '()'

    def make_depth_transform() -> transforms.Compose:
        return transforms.Compose([transforms.ToTensor(), lambda x: 255.0 * x[:3], transforms.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)), CenterPadding(multiple=14)])
    transform = make_depth_transform()
    original_pixel_values = transform(image).unsqueeze(0)
    return original_pixel_values