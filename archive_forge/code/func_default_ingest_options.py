import copy
from typing import Dict, List, Literal, Optional, Union
import ray
from ray.actor import ActorHandle
from ray.data import DataIterator, Dataset, ExecutionOptions, NodeIdStr
from ray.data._internal.execution.interfaces.execution_options import ExecutionResources
from ray.data.preprocessor import Preprocessor
from ray.train.constants import TRAIN_DATASET_KEY  # noqa
from ray.util.annotations import DeveloperAPI, PublicAPI
@staticmethod
def default_ingest_options() -> ExecutionOptions:
    """The default Ray Data options used for data ingest.

        By default, output locality is enabled, which means that Ray Data will try to
        place tasks on the node the data is consumed. The remaining configurations are
        carried over from what is already set in DataContext.
        """
    ctx = ray.data.DataContext.get_current()
    return ExecutionOptions(locality_with_output=True, resource_limits=ctx.execution_options.resource_limits, preserve_order=ctx.execution_options.preserve_order, verbose_progress=ctx.execution_options.verbose_progress)