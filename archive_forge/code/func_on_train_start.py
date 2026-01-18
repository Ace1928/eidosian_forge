import copy
from datetime import datetime
from typing import Callable, Dict, Optional, Union
from packaging import version
import wandb
from wandb.sdk.lib import telemetry
def on_train_start(self, trainer: TRAINER_TYPE):
    with telemetry.context(run=wandb.run) as tel:
        tel.feature.ultralytics_yolov8 = True
    wandb.config.train = vars(trainer.args)
    self.run_id = wandb.run.id