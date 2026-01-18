import os
from typing import Optional, Union
import numpy as np
from huggingface_hub import hf_hub_download
from ... import AutoTokenizer
from ...utils import logging
def search_batched(self, question_projection):
    retrieved_block_ids, _ = self.searcher.search_batched(question_projection.detach().cpu())
    return retrieved_block_ids.astype('int64')