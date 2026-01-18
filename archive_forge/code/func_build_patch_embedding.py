import math
from dataclasses import dataclass
from enum import Enum
import torch
def build_patch_embedding(config: PatchEmbeddingConfig):
    if not isinstance(config, PatchEmbeddingConfig):
        config = PatchEmbeddingConfig(**config)
    if config.pool_type == PoolType.Conv2D:
        pool = torch.nn.Conv2d(config.in_channels, config.out_channels, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding)
    else:
        raise NotImplementedError
    return torch.nn.Sequential(ConditionalReshape(), pool, PatchToSequence())