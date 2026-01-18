import os
import torch
import warnings
from typing import List
from string import Template
from enum import Enum
def get_total_length(self, data):
    prefix = 'Is the <Text> field safe or unsafe '
    input_sample = '<Text> {output} <Context> '.format(**data[0])
    return len(self.tokenizer(prefix + input_sample)['input_ids'])