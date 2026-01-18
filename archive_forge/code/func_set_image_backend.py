import os
import warnings
from modulefinder import Module
import torch
from torchvision import _meta_registrations, datasets, io, models, ops, transforms, utils
from .extension import _HAS_OPS
def set_image_backend(backend):
    """
    Specifies the package used to load images.

    Args:
        backend (string): Name of the image backend. one of {'PIL', 'accimage'}.
            The :mod:`accimage` package uses the Intel IPP library. It is
            generally faster than PIL, but does not support as many operations.
    """
    global _image_backend
    if backend not in ['PIL', 'accimage']:
        raise ValueError(f"Invalid backend '{backend}'. Options are 'PIL' and 'accimage'")
    _image_backend = backend