import collections
from enum import Enum
import json
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
def tag_air_trainer(trainer: 'BaseTrainer'):
    from ray.train.trainer import BaseTrainer
    assert isinstance(trainer, BaseTrainer)
    trainer_name = _find_class_name(trainer, 'ray.train', AIR_TRAINERS)
    record_extra_usage_tag(TagKey.AIR_TRAINER, trainer_name)