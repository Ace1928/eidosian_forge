from dataclasses import dataclass
from os import PathLike
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
def embed_text(self, x: Tensor) -> Tensor:
    positional_embedding = self.position_embeddings(self.get_position_ids(x))
    x = self.word_embeddings(x) + positional_embedding
    return self.dropout(self.layer_norm(x))