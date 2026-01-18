import contextlib
import io
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import get_full_repo_name
from packaging import version
from .debug_utils import DebugOption
from .trainer_utils import (
from .utils import (
from .utils.generic import strtobool
from .utils.import_utils import is_optimum_neuron_available
def set_push_to_hub(self, model_id: str, strategy: Union[str, HubStrategy]='every_save', token: Optional[str]=None, private_repo: bool=False, always_push: bool=False):
    """
        A method that regroups all arguments linked to synchronizing checkpoints with the Hub.

        <Tip>

        Calling this method will set `self.push_to_hub` to `True`, which means the `output_dir` will begin a git
        directory synced with the repo (determined by `model_id`) and the content will be pushed each time a save is
        triggered (depending on`self.save_strategy`). Calling [`~Trainer.save_model`] will also trigger a push.

        </Tip>

        Args:
            model_id (`str`):
                The name of the repository to keep in sync with the local *output_dir*. It can be a simple model ID in
                which case the model will be pushed in your namespace. Otherwise it should be the whole repository
                name, for instance `"user_name/model"`, which allows you to push to an organization you are a member of
                with `"organization_name/model"`.
            strategy (`str` or [`~trainer_utils.HubStrategy`], *optional*, defaults to `"every_save"`):
                Defines the scope of what is pushed to the Hub and when. Possible values are:

                - `"end"`: push the model, its configuration, the tokenizer (if passed along to the [`Trainer`]) and a
                draft of a model card when the [`~Trainer.save_model`] method is called.
                - `"every_save"`: push the model, its configuration, the tokenizer (if passed along to the [`Trainer`])
                  and
                a draft of a model card each time there is a model save. The pushes are asynchronous to not block
                training, and in case the save are very frequent, a new push is only attempted if the previous one is
                finished. A last push is made with the final model at the end of training.
                - `"checkpoint"`: like `"every_save"` but the latest checkpoint is also pushed in a subfolder named
                last-checkpoint, allowing you to resume training easily with
                `trainer.train(resume_from_checkpoint="last-checkpoint")`.
                - `"all_checkpoints"`: like `"checkpoint"` but all checkpoints are pushed like they appear in the
                  output
                folder (so you will get one checkpoint folder per folder in your final repository)

            token (`str`, *optional*):
                The token to use to push the model to the Hub. Will default to the token in the cache folder obtained
                with `huggingface-cli login`.
            private_repo (`bool`, *optional*, defaults to `False`):
                If True, the Hub repo will be set to private.
            always_push (`bool`, *optional*, defaults to `False`):
                Unless this is `True`, the `Trainer` will skip pushing a checkpoint when the previous push is not
                finished.

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_push_to_hub("me/awesome-model")
        >>> args.hub_model_id
        'me/awesome-model'
        ```
        """
    self.push_to_hub = True
    self.hub_model_id = model_id
    self.hub_strategy = HubStrategy(strategy)
    self.hub_token = token
    self.hub_private_repo = private_repo
    self.hub_always_push = always_push
    return self