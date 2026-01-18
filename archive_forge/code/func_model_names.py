import dataclasses
import json
import warnings
from dataclasses import dataclass, field
from time import time
from typing import List
from ..utils import logging
@property
def model_names(self) -> List[str]:
    if len(self.models) <= 0:
        raise ValueError("Please make sure you provide at least one model name / model identifier, *e.g.* `--models google-bert/bert-base-cased` or `args.models = ['google-bert/bert-base-cased'].")
    return self.models