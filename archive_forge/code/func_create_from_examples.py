import csv
import dataclasses
import json
from dataclasses import dataclass
from typing import List, Optional, Union
from ...utils import is_tf_available, is_torch_available, logging
@classmethod
def create_from_examples(cls, texts_or_text_and_labels, labels=None, **kwargs):
    processor = cls(**kwargs)
    processor.add_examples(texts_or_text_and_labels, labels=labels)
    return processor