import collections.abc
import math
from typing import Dict, List, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from ...utils.backbone_utils import load_backbone
from .configuration_vit_hybrid import ViTHybridConfig
def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
    num_patches = embeddings.shape[1] - 1
    num_positions = self.position_embeddings.shape[1] - 1
    if num_patches == num_positions and height == width:
        return self.position_embeddings
    class_pos_embed = self.position_embeddings[:, 0]
    patch_pos_embed = self.position_embeddings[:, 1:]
    dim = embeddings.shape[-1]
    height = height // self.config.patch_size
    width = width // self.config.patch_size
    height, width = (height + 0.1, width + 0.1)
    patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
    patch_pos_embed = nn.functional.interpolate(patch_pos_embed, scale_factor=(height / math.sqrt(num_positions), width / math.sqrt(num_positions)), mode='bicubic', align_corners=False)
    if int(height) != patch_pos_embed.shape[-2] or int(width) != patch_pos_embed.shape[-1]:
        raise ValueError(f'Invalid height or width: {height}, {width}')
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)