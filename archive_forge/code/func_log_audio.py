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
def log_audio(self, key: str, audios: List[Any], step: Optional[int]=None, **kwargs: Any) -> None:
    """Log audios (numpy arrays, or file paths).

        Args:
            key: The key to be used for logging the audio files
            audios: The list of audio file paths, or numpy arrays to be logged
            step: The step number to be used for logging the audio files
            \\**kwargs: Optional kwargs are lists passed to each ``Wandb.Audio`` instance (ex: caption, sample_rate).

        Optional kwargs are lists passed to each audio (ex: caption, sample_rate).

        """
    if not isinstance(audios, list):
        raise TypeError(f'Expected a list as "audios", found {type(audios)}')
    n = len(audios)
    for k, v in kwargs.items():
        if len(v) != n:
            raise ValueError(f'Expected {n} items but only found {len(v)} for {k}')
    kwarg_list = [{k: kwargs[k][i] for k in kwargs} for i in range(n)]
    metrics = {key: [wandb.Audio(audio, **kwarg) for audio, kwarg in zip(audios, kwarg_list)]}
    self.log_metrics(metrics, step)