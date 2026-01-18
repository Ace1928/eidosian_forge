from dataclasses import dataclass
from os import PathLike
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
def get_matching_scores(self, embeddings: Tensor) -> Tensor:
    """Computes the probability that there is a match between images and texts based on their multimodal embeddings

        :param embeddings: multimodal joint embeddings
        """
    return self.text_encoder.forward_matching(embeddings)