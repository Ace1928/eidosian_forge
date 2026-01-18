import logging
import os
import random
import time
import urllib
from typing import Any, Callable, Optional, Sized, Tuple, Union
from urllib.error import HTTPError
from warnings import warn
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from lightning_fabric.utilities.imports import _IS_WINDOWS
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.imports import _TORCHVISION_AVAILABLE
@staticmethod
def normalize_tensor(tensor: Tensor, mean: Union[int, float]=0.0, std: Union[int, float]=1.0) -> Tensor:
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    return tensor.sub(mean).div(std)