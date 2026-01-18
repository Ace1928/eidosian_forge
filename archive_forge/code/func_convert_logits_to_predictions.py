import collections
import datetime
import enum
import itertools
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, Generator, List, Optional, Text, Tuple, Union
import numpy as np
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...tokenization_utils_base import (
from ...utils import ExplicitEnum, PaddingStrategy, TensorType, add_end_docstrings, is_pandas_available, logging
def convert_logits_to_predictions(self, data, logits, logits_agg=None, cell_classification_threshold=0.5):
    """
        Converts logits of [`TapasForQuestionAnswering`] to actual predicted answer coordinates and optional
        aggregation indices.

        The original implementation, on which this function is based, can be found
        [here](https://github.com/google-research/tapas/blob/4908213eb4df7aa988573350278b44c4dbe3f71b/tapas/experiments/prediction_utils.py#L288).

        Args:
            data (`dict`):
                Dictionary mapping features to actual values. Should be created using [`TapasTokenizer`].
            logits (`torch.Tensor` or `tf.Tensor` of shape `(batch_size, sequence_length)`):
                Tensor containing the logits at the token level.
            logits_agg (`torch.Tensor` or `tf.Tensor` of shape `(batch_size, num_aggregation_labels)`, *optional*):
                Tensor containing the aggregation logits.
            cell_classification_threshold (`float`, *optional*, defaults to 0.5):
                Threshold to be used for cell selection. All table cells for which their probability is larger than
                this threshold will be selected.

        Returns:
            `tuple` comprising various elements depending on the inputs:

            - predicted_answer_coordinates (`List[List[[tuple]]` of length `batch_size`): Predicted answer coordinates
              as a list of lists of tuples. Each element in the list contains the predicted answer coordinates of a
              single example in the batch, as a list of tuples. Each tuple is a cell, i.e. (row index, column index).
            - predicted_aggregation_indices (`List[int]`of length `batch_size`, *optional*, returned when
              `logits_aggregation` is provided): Predicted aggregation operator indices of the aggregation head.
        """
    logits = logits.numpy()
    if logits_agg is not None:
        logits_agg = logits_agg.numpy()
    data = {key: value.numpy() for key, value in data.items() if key != 'training'}
    logits[logits < -88.7] = -88.7
    probabilities = 1 / (1 + np.exp(-logits)) * data['attention_mask']
    token_types = ['segment_ids', 'column_ids', 'row_ids', 'prev_labels', 'column_ranks', 'inv_column_ranks', 'numeric_relations']
    input_ids = data['input_ids']
    segment_ids = data['token_type_ids'][:, :, token_types.index('segment_ids')]
    row_ids = data['token_type_ids'][:, :, token_types.index('row_ids')]
    column_ids = data['token_type_ids'][:, :, token_types.index('column_ids')]
    num_batch = input_ids.shape[0]
    predicted_answer_coordinates = []
    for i in range(num_batch):
        probabilities_example = probabilities[i].tolist()
        segment_ids_example = segment_ids[i]
        row_ids_example = row_ids[i]
        column_ids_example = column_ids[i]
        max_width = column_ids_example.max()
        max_height = row_ids_example.max()
        if max_width == 0 and max_height == 0:
            continue
        cell_coords_to_prob = self._get_mean_cell_probs(probabilities_example, segment_ids_example.tolist(), row_ids_example.tolist(), column_ids_example.tolist())
        answer_coordinates = []
        for col in range(max_width):
            for row in range(max_height):
                cell_prob = cell_coords_to_prob.get((col, row), None)
                if cell_prob is not None:
                    if cell_prob > cell_classification_threshold:
                        answer_coordinates.append((row, col))
        answer_coordinates = sorted(answer_coordinates)
        predicted_answer_coordinates.append(answer_coordinates)
    output = (predicted_answer_coordinates,)
    if logits_agg is not None:
        predicted_aggregation_indices = logits_agg.argmax(axis=-1)
        output = (predicted_answer_coordinates, predicted_aggregation_indices.tolist())
    return output