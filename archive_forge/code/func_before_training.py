import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, cast
import xgboost as xgb  # type: ignore
from xgboost import Booster
import wandb
from wandb.sdk.lib import telemetry as wb_telemetry
def before_training(self, model: Booster) -> Booster:
    """Run before training is finished."""
    config = model.save_config()
    wandb.config.update(json.loads(config))
    return model