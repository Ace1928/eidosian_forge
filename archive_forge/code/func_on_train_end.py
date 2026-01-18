import functools
import importlib.metadata
import importlib.util
import json
import numbers
import os
import pickle
import shutil
import sys
import tempfile
from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union
import numpy as np
from .. import __version__ as version
from ..utils import flatten_dict, is_datasets_available, is_pandas_available, is_torch_available, logging
from ..trainer_callback import ProgressCallback, TrainerCallback  # noqa: E402
from ..trainer_utils import PREFIX_CHECKPOINT_DIR, BestRun, IntervalStrategy  # noqa: E402
from ..training_args import ParallelMode  # noqa: E402
from ..utils import ENV_VARS_TRUE_VALUES, is_torch_tpu_available  # noqa: E402
def on_train_end(self, args, state, control, **kwargs):
    if self._initialized and state.is_world_process_zero:
        from transformers.trainer import Trainer
        if self._log_model is True:
            fake_trainer = Trainer(args=args, model=kwargs.get('model'), tokenizer=kwargs.get('tokenizer'))
            name = 'best' if args.load_best_model_at_end else 'last'
            output_dir = os.path.join(args.output_dir, name)
            fake_trainer.save_model(output_dir)
            self.live.log_artifact(output_dir, name=name, type='model', copy=True)
        self.live.end()