import copy
from datetime import datetime
from typing import Callable, Dict, Optional, Union
from packaging import version
import wandb
from wandb.sdk.lib import telemetry
def add_wandb_callback(model: YOLO, epoch_logging_interval: int=1, enable_model_checkpointing: bool=False, enable_train_validation_logging: bool=True, enable_validation_logging: bool=True, enable_prediction_logging: bool=True, max_validation_batches: Optional[int]=1, visualize_skeleton: Optional[bool]=True):
    """Function to add the `WandBUltralyticsCallback` callback to the `YOLO` model.

    Example:
        ```python
        from ultralytics.yolo.engine.model import YOLO
        from wandb.yolov8 import add_wandb_callback

        # initialize YOLO model
        model = YOLO("yolov8n.pt")

        # add wandb callback
        add_wandb_callback(model, max_validation_batches=2, enable_model_checkpointing=True)

        # train
        model.train(data="coco128.yaml", epochs=5, imgsz=640)

        # validate
        model.val()

        # perform inference
        model(["img1.jpeg", "img2.jpeg"])
        ```

    Arguments:
        model: (ultralytics.yolo.engine.model.YOLO) YOLO Model of type
            `ultralytics.yolo.engine.model.YOLO`.
        epoch_logging_interval: (int) interval to log the prediction visualizations
            during training.
        enable_model_checkpointing: (bool) enable logging model checkpoints as
            artifacts at the end of eveny epoch if set to `True`.
        enable_train_validation_logging: (bool) enable logging the predictions and
            ground-truths as interactive image overlays on the images from
            the validation dataloader to a `wandb.Table` along with
            mean-confidence of the predictions per-class at the end of each
            training epoch.
        enable_validation_logging: (bool) enable logging the predictions and
            ground-truths as interactive image overlays on the images from the
            validation dataloader to a `wandb.Table` along with
            mean-confidence of the predictions per-class at the end of
            validation.
        enable_prediction_logging: (bool) enable logging the predictions and
            ground-truths as interactive image overlays on the images from the
            validation dataloader to a `wandb.Table` along with mean-confidence
            of the predictions per-class at the end of each prediction.
        max_validation_batches: (Optional[int]) maximum number of validation batches to log to
            a table per epoch.
        visualize_skeleton: (Optional[bool]) visualize pose skeleton by drawing lines connecting
            keypoints for human pose.

    Returns:
        An instance of `ultralytics.yolo.engine.model.YOLO` with the `WandBUltralyticsCallback`.
    """
    if RANK in [-1, 0]:
        wandb_callback = WandBUltralyticsCallback(copy.deepcopy(model), epoch_logging_interval, max_validation_batches, enable_model_checkpointing, visualize_skeleton)
        callbacks = wandb_callback.callbacks
        if not enable_train_validation_logging:
            _ = callbacks.pop('on_fit_epoch_end')
            _ = callbacks.pop('on_train_end')
        if not enable_validation_logging:
            _ = callbacks.pop('on_val_start')
            _ = callbacks.pop('on_val_end')
        if not enable_prediction_logging:
            _ = callbacks.pop('on_predict_start')
            _ = callbacks.pop('on_predict_end')
        for event, callback_fn in callbacks.items():
            model.add_callback(event, callback_fn)
    else:
        wandb.termerror('The RANK of the process to add the callbacks was neither 0 or -1. No Weights & Biases callbacks were added to this instance of the YOLO model.')
    return model