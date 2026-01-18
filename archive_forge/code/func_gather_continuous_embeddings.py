from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...models.auto.modeling_auto import AutoModelForCausalLM
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_fuyu import FuyuConfig
def gather_continuous_embeddings(self, word_embeddings: torch.Tensor, continuous_embeddings: List[torch.Tensor], image_patch_input_indices: torch.Tensor) -> torch.Tensor:
    """This function places the continuous_embeddings into the word_embeddings at the locations
        indicated by image_patch_input_indices. Different batch elements can have different numbers of continuous
        embeddings.

        Args:
            word_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Tensor of word embeddings.
            continuous_embeddings (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`):
                Tensor of continuous embeddings. The length of the list is the batch size. Each entry is shape
                [num_image_embeddings, hidden], and num_image_embeddings needs to match the number of non-negative
                indices in image_patch_input_indices for that batch element.
            image_patch_input_indices (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Tensor of indices of the image patches in the input_ids tensor.
        """
    if not word_embeddings.shape[0] == len(continuous_embeddings):
        raise ValueError(f'Batch sizes must match! Got len(continuous_embeddings)={len(continuous_embeddings)!r} and word_embeddings.shape[0]={word_embeddings.shape[0]!r}')
    output_embeddings = word_embeddings.clone()
    for batch_idx in range(word_embeddings.shape[0]):
        dst_indices = torch.nonzero(image_patch_input_indices[batch_idx] >= 0, as_tuple=True)[0]
        src_indices = image_patch_input_indices[batch_idx][dst_indices]
        if src_indices.shape[0] > continuous_embeddings[batch_idx].shape[0]:
            raise ValueError(f'Number of continuous embeddings continuous_embeddings[batch_idx].shape={continuous_embeddings[batch_idx].shape!r} does not match number of continuous token ids src_indices.shape={src_indices.shape!r} in batch element {batch_idx}.')
        output_embeddings[batch_idx, dst_indices] = continuous_embeddings[batch_idx][src_indices]
    return output_embeddings