import logging
import os
from argparse import Namespace
from typing import Any, Dict, Optional, Union
from typing_extensions import override
from lightning_fabric.loggers.csv_logs import CSVLogger as FabricCSVLogger
from lightning_fabric.loggers.csv_logs import _ExperimentWriter as _FabricExperimentWriter
from lightning_fabric.loggers.logger import rank_zero_experiment
from lightning_fabric.utilities.logger import _convert_params
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
def log_hparams(self, params: Dict[str, Any]) -> None:
    """Record hparams."""
    self.hparams.update(params)