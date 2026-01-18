import math
from typing import Dict, List, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import (
from ...image_utils import (
from ...utils import (
def preprocess_with_tokenizer_info(self, image_input: 'torch.Tensor', image_present: 'torch.Tensor', image_unpadded_h: 'torch.Tensor', image_unpadded_w: 'torch.Tensor', image_placeholder_id: int, image_newline_id: int, variable_sized: bool, patch_size: Optional[Dict[str, int]]=None) -> FuyuBatchFeature:
    """Process images for model input. In particular, variable-sized images are handled here.

        Args:
            image_input (`torch.Tensor` of shape [batch_size, subsequence_size, num_channels, height, width]):
                Tensor of images padded to model input size.
            image_present (`torch.Tensor` of shape [batch_size, subsequence_size, num_images]):
                Tensor of 1s and 0s indicating whether an image is present.
            image_unpadded_h (`torch.Tensor` of shape [batch_size, subsequence_size]):
                Tensor of unpadded image heights.
            image_unpadded_w (`torch.Tensor` of shape [batch_size, subsequence_size]):
                Tensor of unpadded image widths.
            image_placeholder_id (int):
                The id of the image placeholder token. Comes from an associated tokenizer.
            image_newline_id (int):
                The id of the image newline token. Comes from an associated tokenizer.
            variable_sized (bool):
                Whether to process images as variable-sized.
            patch_size (`Dict[str, int]`, *optional*, defaults to `self.patch_size`):
                Size of the patches.
        """
    requires_backends(self, ['torch'])
    patch_size = patch_size if patch_size is not None else self.patch_size
    patch_height, patch_width = (patch_size['height'], patch_size['width'])
    images: List[List[torch.Tensor]] = []
    batch_image_patches: List[List[torch.Tensor]] = []
    batch_image_input_ids: List[List[torch.Tensor]] = []
    for batch_index in range(image_input.shape[0]):
        image_input_ids = []
        image_patches = []
        for subseq_index in range(image_input.shape[1]):
            if image_present[batch_index, subseq_index]:
                image = image_input[batch_index, subseq_index]
                image_height, image_width = (image.shape[1], image.shape[2])
                if variable_sized:
                    new_h = min(image_height, math.ceil(image_unpadded_h[batch_index, subseq_index] / patch_height) * patch_height)
                    new_w = min(image_width, math.ceil(image_unpadded_w[batch_index, subseq_index] / patch_width) * patch_width)
                    image = image[:, :new_h, :new_w]
                    image_height, image_width = (new_h, new_w)
                num_patches = self.get_num_patches(image_height=image_height, image_width=image_width)
                tensor_of_image_ids = torch.full([num_patches], image_placeholder_id, dtype=torch.int32, device=image_input.device)
                patches = self.patchify_image(image=image.unsqueeze(0)).squeeze(0)
                assert num_patches == patches.shape[0]
                if variable_sized:
                    tensor_of_image_ids = tensor_of_image_ids.reshape(-1, image_width // patch_width)
                    newline_ids = torch.full([tensor_of_image_ids.shape[0], 1], image_newline_id, dtype=torch.int32, device=image_input.device)
                    tensor_of_image_ids = torch.cat([tensor_of_image_ids, newline_ids], dim=1)
                    tensor_of_image_ids = tensor_of_image_ids.reshape(-1)
                images.append([image])
                image_input_ids.append(tensor_of_image_ids)
                image_patches.append(patches)
            else:
                image_input_ids.append(torch.tensor([], dtype=torch.int32, device=image_input.device))
        batch_image_input_ids.append(image_input_ids)
        batch_image_patches.append(image_patches)
    image_patch_indices_per_batch: List[List[torch.Tensor]] = []
    image_patch_indices_per_subsequence: List[List[torch.Tensor]] = []
    for sample_image_input_ids in batch_image_input_ids:
        index_offset = 0
        per_batch_indices = []
        per_subsequence_indices = []
        for subseq_image_input_ids in sample_image_input_ids:
            patches_mask = subseq_image_input_ids == image_placeholder_id
            num_patches = torch.count_nonzero(patches_mask)
            indices = torch.arange(num_patches, dtype=torch.int64, device=subseq_image_input_ids.device).type_as(subseq_image_input_ids)
            indices_in_stream_per_batch = torch.full_like(subseq_image_input_ids, -1)
            indices_in_stream_per_subsequence = torch.full_like(subseq_image_input_ids, -1)
            patches_inds = torch.nonzero(patches_mask, as_tuple=True)[0]
            indices_in_stream_per_batch[patches_inds] = indices + index_offset
            indices_in_stream_per_subsequence[patches_inds] = indices
            per_batch_indices.append(indices_in_stream_per_batch)
            per_subsequence_indices.append(indices_in_stream_per_subsequence)
            index_offset += num_patches
        image_patch_indices_per_batch.append(per_batch_indices)
        image_patch_indices_per_subsequence.append(per_subsequence_indices)
    return FuyuBatchFeature(data={'images': images, 'image_input_ids': batch_image_input_ids, 'image_patches': batch_image_patches, 'image_patch_indices_per_batch': image_patch_indices_per_batch, 'image_patch_indices_per_subsequence': image_patch_indices_per_subsequence})