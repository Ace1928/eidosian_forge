import os
from typing import Any, Dict, Union
import lightgbm
import lightgbm_ray
import xgboost_ray
from lightgbm_ray.tune import TuneReportCheckpointCallback
from ray.train import Checkpoint
from ray.train.gbdt_trainer import GBDTTrainer
from ray.train.lightgbm import LightGBMCheckpoint
from ray.util.annotations import PublicAPI
def preprocess_datasets(self) -> None:
    super().preprocess_datasets()
    if Version(xgboost_ray.__version__) < Version('0.1.16'):
        self._repartition_datasets_to_match_num_actors()