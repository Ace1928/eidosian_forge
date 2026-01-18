import copy
from dataclasses import dataclass, field
from typing import ClassVar, Dict
from ..features import Audio, ClassLabel, Features
from .base import TaskTemplate
def align_with_features(self, features):
    if self.label_column not in features:
        raise ValueError(f'Column {self.label_column} is not present in features.')
    if not isinstance(features[self.label_column], ClassLabel):
        raise ValueError(f'Column {self.label_column} is not a ClassLabel.')
    task_template = copy.deepcopy(self)
    label_schema = self.label_schema.copy()
    label_schema['labels'] = features[self.label_column]
    task_template.__dict__['label_schema'] = label_schema
    return task_template