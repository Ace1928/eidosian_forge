import logging
import os
import types
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Optional, Union
import torch
from packaging.version import parse
from ..utils import check_if_pytorch_greater, is_accelerate_available, recurse_getattr, recurse_setattr
from .models import BetterTransformerManager
def raise_save_or_push_incompatible(*_, **__):
    """
    Simply raise an error if the user tries to save or push a model that is not compatible with
    `BetterTransformer` and needs to be reverted to the original model before calling these
    functions.
    """
    raise ValueError('You are trying to save or push a model that has been converted with `BetterTransformer`.', ' Please revert the model to its original state before calling `save_pretrained` or `push_to_hub`.', ' By calling model = BetterTransformer.reverse(model) before saving or pushing.')