import copy
from dataclasses import dataclass, field
from typing import ClassVar, Dict
from ..features import Audio, ClassLabel, Features
from .base import TaskTemplate
@property
def column_mapping(self) -> Dict[str, str]:
    return {self.audio_column: 'audio', self.label_column: 'labels'}