import collections
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_flava import (
def get_codebook_indices(self, pixel_values: torch.Tensor) -> torch.Tensor:
    '\n        Args:\n            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n                Pixel values. Codebook pixel values can be obtained using [`AutoImageProcessor`] by passing\n                `return_codebook_pixels=True`. See [`FlavaImageProcessor.__call__`] for details.\n\n        Examples:\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoImageProcessor, FlavaImageCodebook\n\n        >>> model = FlavaImageCodebook.from_pretrained("{0}")\n        >>> image_processor = AutoImageProcessor.from_pretrained("{0}")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> inputs = image_processor([image], return_codebook_pixels=True, return_tensors="pt")\n        >>> inputs = dict(pixel_values=inputs.codebook_pixel_values)\n\n        >>> outputs = model.get_codebook_indices(**inputs)\n        ```\n        '.format(_CHECKPOINT_FOR_CODEBOOK_DOC)
    z_logits = self.blocks(pixel_values)
    return torch.argmax(z_logits, axis=1)