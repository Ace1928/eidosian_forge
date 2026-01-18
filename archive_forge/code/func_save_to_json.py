import dataclasses
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np
from tqdm.auto import tqdm
from .trainer_utils import IntervalStrategy, has_length
from .training_args import TrainingArguments
from .utils import logging
def save_to_json(self, json_path: str):
    """Save the content of this instance in JSON format inside `json_path`."""
    json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + '\n'
    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(json_string)