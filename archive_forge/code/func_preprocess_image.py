from os import PathLike
from typing import Dict, List, Union
import torch
from PIL.Image import Image
from tokenizers import Tokenizer
from torch import Tensor
from torchvision.transforms import (
def preprocess_image(self, images: Union[Image, List[Image]]) -> Tensor:
    """Transforms one or more Pillow images into Torch Tensors.

        :param images: image or list of images to preprocess
        """
    if isinstance(images, list):
        batch_images = torch.empty((len(images), 3, self._image_size, self._image_size), dtype=torch.float32)
        for i, image in enumerate(images):
            batch_images[i] = self._image_transform(image)
    else:
        batch_images = self._image_transform(images).unsqueeze(0)
    return batch_images