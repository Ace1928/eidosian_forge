import itertools
import logging
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple
import mlflow
from packaging.version import Version  # type: ignore
import wandb
from wandb import Artifact
from .internals import internal
from .internals.util import Namespace, for_each
def collect_runs(self, *, limit: Optional[int]=None) -> Iterable[MlflowRun]:
    if mlflow_version < Version('1.28.0'):
        experiments = self.mlflow_client.list_experiments()
    else:
        experiments = self.mlflow_client.search_experiments()

    def _runs():
        for exp in experiments:
            for run in self.mlflow_client.search_runs(exp.experiment_id):
                yield MlflowRun(run, self.mlflow_client)
    runs = itertools.islice(_runs(), limit)
    yield from runs