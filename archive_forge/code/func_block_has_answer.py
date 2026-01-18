import os
from typing import Optional, Union
import numpy as np
from huggingface_hub import hf_hub_download
from ... import AutoTokenizer
from ...utils import logging
def block_has_answer(self, concat_inputs, answer_ids):
    """check if retrieved_blocks has answers."""
    has_answers = []
    start_pos = []
    end_pos = []
    max_answers = 0
    for input_id in concat_inputs.input_ids:
        input_id_list = input_id.tolist()
        first_sep_idx = input_id_list.index(self.tokenizer.sep_token_id)
        second_sep_idx = first_sep_idx + 1 + input_id_list[first_sep_idx + 1:].index(self.tokenizer.sep_token_id)
        start_pos.append([])
        end_pos.append([])
        for answer in answer_ids:
            for idx in range(first_sep_idx + 1, second_sep_idx):
                if answer[0] == input_id_list[idx]:
                    if input_id_list[idx:idx + len(answer)] == answer:
                        start_pos[-1].append(idx)
                        end_pos[-1].append(idx + len(answer) - 1)
        if len(start_pos[-1]) == 0:
            has_answers.append(False)
        else:
            has_answers.append(True)
            if len(start_pos[-1]) > max_answers:
                max_answers = len(start_pos[-1])
    for start_pos_, end_pos_ in zip(start_pos, end_pos):
        if len(start_pos_) < max_answers:
            padded = [-1] * (max_answers - len(start_pos_))
            start_pos_ += padded
            end_pos_ += padded
    return (has_answers, start_pos, end_pos)