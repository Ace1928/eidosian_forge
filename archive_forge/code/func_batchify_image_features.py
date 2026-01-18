from typing import Dict, List, Tuple
import torch
from .modules import ImageSeq2seqModel, FusionType
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.dict import DictionaryAgent
from parlai.core.torch_agent import Batch
from parlai.core.torch_image_agent import TorchImageAgent
def batchify_image_features(self, batch: Batch) -> Batch:
    """
        Format and return the batched image features.

        Image features represented by tensors will set to the right type.
        """
    if type(batch.image) == list and any((b is not None for b in batch.image)):
        images = []
        for img in batch.image:
            if isinstance(img, torch.Tensor):
                img = self._process_image_features(img)
            images.append(img)
        batch.image = images
    else:
        images = [None] * len(batch.valid_indices)
        batch.image = images
    return batch