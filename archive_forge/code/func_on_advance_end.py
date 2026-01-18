import logging
from typing import Any, Dict, List, Optional, Union
import torch
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.data import _set_sampler_epoch, sized_len
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.loops import _Loop
from pytorch_lightning.loops.fetchers import _DataFetcher
from pytorch_lightning.loops.progress import _Progress
from pytorch_lightning.loops.training_epoch_loop import _TrainingEpochLoop
from pytorch_lightning.loops.utilities import _is_max_limit_reached, _select_data_fetcher
from pytorch_lightning.trainer import call
from pytorch_lightning.trainer.connectors.data_connector import (
from pytorch_lightning.trainer.connectors.logger_connector.result import _ResultCollection
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.utilities.combined_loader import _SUPPORTED_MODES, CombinedLoader
from pytorch_lightning.utilities.data import has_len_all_ranks
from pytorch_lightning.utilities.exceptions import MisconfigurationException, SIGTERMException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_debug, rank_zero_info, rank_zero_warn
def on_advance_end(self) -> None:
    trainer = self.trainer
    trainer._logger_connector.epoch_end_reached()
    self.epoch_progress.increment_processed()
    call._call_callback_hooks(trainer, 'on_train_epoch_end', monitoring_callbacks=False)
    call._call_lightning_module_hook(trainer, 'on_train_epoch_end')
    call._call_callback_hooks(trainer, 'on_train_epoch_end', monitoring_callbacks=True)
    trainer._logger_connector.on_epoch_end()
    if self.epoch_loop._num_ready_batches_reached():
        self.epoch_loop.update_lr_schedulers('epoch', update_plateau_schedulers=not self.restarting)
    self.epoch_loop._batches_that_stepped -= 1
    trainer._logger_connector.update_train_epoch_metrics()
    self.epoch_loop._batches_that_stepped += 1
    self.epoch_progress.increment_completed()
    if trainer.received_sigterm:
        raise SIGTERMException