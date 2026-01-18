import argparse
import collections
from pathlib import Path
import torch
from torch.serialization import default_restore_location
from transformers import BertConfig, DPRConfig, DPRContextEncoder, DPRQuestionEncoder, DPRReader
def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    print(f'Reading saved model from {model_file}')
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    return CheckpointState(**state_dict)