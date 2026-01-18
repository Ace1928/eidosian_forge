import os
import warnings
from modulefinder import Module
import torch
from torchvision import _meta_registrations, datasets, io, models, ops, transforms, utils
from .extension import _HAS_OPS
def get_video_backend():
    """
    Returns the currently active video backend used to decode videos.

    Returns:
        str: Name of the video backend. one of {'pyav', 'video_reader'}.
    """
    return _video_backend