import contextlib
import gc
import tempfile
from collections import OrderedDict
from unittest.mock import patch, MagicMock
import pytest
import ray
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
import vllm
from vllm.config import LoRAConfig
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.parallel_utils.parallel_state import (
@pytest.fixture
def llama_2_7b_model_extra_embeddings(llama_2_7b_engine_extra_embeddings) -> nn.Module:
    yield llama_2_7b_engine_extra_embeddings.driver_worker.model_runner.model