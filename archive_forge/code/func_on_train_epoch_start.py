from typing import Any, Callable, Dict, List, Optional
from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.v8.classify.train import ClassificationTrainer
import wandb
from wandb.sdk.lib import telemetry
def on_train_epoch_start(self, trainer: BaseTrainer) -> None:
    """On train epoch start we only log epoch number to the Weights & Biases run."""
    assert self.run is not None
    self.run.log({'epoch': trainer.epoch + 1})