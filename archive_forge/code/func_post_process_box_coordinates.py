import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, TruncationStrategy
from ...utils import TensorType, is_torch_available, logging, requires_backends
def post_process_box_coordinates(self, outputs, target_sizes=None):
    """
        Transforms raw coordinates detected by [`FuyuForCausalLM`] to the original images' coordinate space.
        Coordinates will be returned in "box" format, with the following pattern:
            `<box>top, left, bottom, right</box>`

        Point coordinates are not supported yet.

        Args:
            outputs ([`GenerateOutput`]):
                Raw outputs from `generate`.
            target_sizes (`torch.Tensor`, *optional*):
                Tensor of shape (batch_size, 2) where each entry is the (height, width) of the corresponding image in
                the batch. If set, found coordinates in the output sequence are rescaled to the target sizes. If left
                to None, coordinates will not be rescaled.

        Returns:
            `GenerateOutput`: Same output type returned by `generate`, with output token ids replaced with
                boxed and possible rescaled coordinates.
        """

    def scale_factor_to_fit(original_size, target_size=None):
        height, width = original_size
        if target_size is None:
            max_height = self.image_processor.size['height']
            max_width = self.image_processor.size['width']
        else:
            max_height, max_width = target_size
        if width <= max_width and height <= max_height:
            return 1.0
        return min(max_height / height, max_width / width)

    def find_delimiters_pair(tokens, start_token, end_token):
        start_id = self.tokenizer.convert_tokens_to_ids(start_token)
        end_id = self.tokenizer.convert_tokens_to_ids(end_token)
        starting_positions = (tokens == start_id).nonzero(as_tuple=True)[0]
        ending_positions = (tokens == end_id).nonzero(as_tuple=True)[0]
        if torch.any(starting_positions) and torch.any(ending_positions):
            return (starting_positions[0], ending_positions[0])
        return (None, None)

    def tokens_to_boxes(tokens, original_size):
        while (pair := find_delimiters_pair(tokens, TOKEN_BBOX_OPEN_STRING, TOKEN_BBOX_CLOSE_STRING)) != (None, None):
            start, end = pair
            if end != start + 5:
                continue
            coords = self.tokenizer.convert_ids_to_tokens(tokens[start + 1:end])
            scale = scale_factor_to_fit(original_size)
            top, left, bottom, right = [2 * int(float(c) / scale) for c in coords]
            replacement = f' {TEXT_REPR_BBOX_OPEN}{top}, {left}, {bottom}, {right}{TEXT_REPR_BBOX_CLOSE}'
            replacement = self.tokenizer.tokenize(replacement)[1:]
            replacement = self.tokenizer.convert_tokens_to_ids(replacement)
            replacement = torch.tensor(replacement).to(tokens)
            tokens = torch.cat([tokens[:start], replacement, tokens[end + 1:]], 0)
        return tokens

    def tokens_to_points(tokens, original_size):
        while (pair := find_delimiters_pair(tokens, TOKEN_POINT_OPEN_STRING, TOKEN_POINT_CLOSE_STRING)) != (None, None):
            start, end = pair
            if end != start + 3:
                continue
            coords = self.tokenizer.convert_ids_to_tokens(tokens[start + 1:end])
            scale = scale_factor_to_fit(original_size)
            x, y = [2 * int(float(c) / scale) for c in coords]
            replacement = f' {TEXT_REPR_POINT_OPEN}{x}, {y}{TEXT_REPR_POINT_CLOSE}'
            replacement = self.tokenizer.tokenize(replacement)[1:]
            replacement = self.tokenizer.convert_tokens_to_ids(replacement)
            replacement = torch.tensor(replacement).to(tokens)
            tokens = torch.cat([tokens[:start], replacement, tokens[end + 1:]], 0)
        return tokens
    if target_sizes is None:
        target_sizes = ((self.image_processor.size['height'], self.image_processor.size['width']),) * len(outputs)
    elif target_sizes.shape[1] != 2:
        raise ValueError('Each element of target_sizes must contain the size (h, w) of each image of the batch')
    if len(outputs) != len(target_sizes):
        raise ValueError('Make sure that you pass in as many target sizes as output sequences')
    results = []
    for seq, size in zip(outputs, target_sizes):
        seq = tokens_to_boxes(seq, size)
        seq = tokens_to_points(seq, size)
        results.append(seq)
    return results