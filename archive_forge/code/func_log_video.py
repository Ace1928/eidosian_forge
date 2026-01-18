import os
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Mapping, Optional, Union
from packaging import version
from typing_extensions import override
import wandb
from wandb import Artifact
from wandb.sdk.lib import RunDisabled, telemetry
from wandb.sdk.wandb_run import Run
@rank_zero_only
def log_video(self, key: str, videos: List[Any], step: Optional[int]=None, **kwargs: Any) -> None:
    """Log videos (numpy arrays, or file paths).

        Args:
            key: The key to be used for logging the video files
            videos: The list of video file paths, or numpy arrays to be logged
            step: The step number to be used for logging the video files
            **kwargs: Optional kwargs are lists passed to each Wandb.Video instance (ex: caption, fps, format).

        Optional kwargs are lists passed to each video (ex: caption, fps, format).

        """
    if not isinstance(videos, list):
        raise TypeError(f'Expected a list as "videos", found {type(videos)}')
    n = len(videos)
    for k, v in kwargs.items():
        if len(v) != n:
            raise ValueError(f'Expected {n} items but only found {len(v)} for {k}')
    kwarg_list = [{k: kwargs[k][i] for k in kwargs} for i in range(n)]
    metrics = {key: [wandb.Video(video, **kwarg) for video, kwarg in zip(videos, kwarg_list)]}
    self.log_metrics(metrics, step)