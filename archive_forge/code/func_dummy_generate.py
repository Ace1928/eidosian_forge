import unittest
from unittest.mock import patch
import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, TextEnvironment, TextHistory
def dummy_generate(histories):
    for i in range(len(histories)):
        histories[i].append_segment('<request><DummyTool>test<call>', torch.tensor([1, 2, 3]), system=False)
    return histories