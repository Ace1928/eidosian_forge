import warnings
from typing import List, Optional, Union
import numpy as np
import PIL
import torch
from diffusers import ConfigMixin
from diffusers.image_processor import VaeImageProcessor as DiffusersVaeImageProcessor
from diffusers.utils.pil_utils import PIL_INTERPOLATION
from PIL import Image
from tqdm.auto import tqdm
@staticmethod
def pt_to_numpy(images: torch.FloatTensor) -> np.ndarray:
    """
        Convert a PyTorch tensor to a NumPy image.
        """
    images = images.cpu().float().numpy()
    return images