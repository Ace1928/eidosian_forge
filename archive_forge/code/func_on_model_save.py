from typing import Any, Callable, Dict, List, Optional
from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.v8.classify.train import ClassificationTrainer
import wandb
from wandb.sdk.lib import telemetry
def on_model_save(self, trainer: BaseTrainer) -> None:
    """On model save we log the model as an artifact to Weights & Biases."""
    assert self.run is not None
    self.run.log_artifact(str(trainer.last), type='model', name=f'{self.run.name}_{trainer.args.task}.pt', aliases=['last', f'epoch_{trainer.epoch + 1}'])