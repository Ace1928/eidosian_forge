from abc import ABCMeta, abstractmethod
from typing import List, Optional
from mlflow.entities import DatasetInput, ViewType
from mlflow.entities.metric import MetricWithRunId
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.utils.annotations import developer_stable
from mlflow.utils.async_logging.async_logging_queue import AsyncLoggingQueue
from mlflow.utils.async_logging.run_operations import RunOperations
def set_tag_async(self, run_id, tag) -> RunOperations:
    """
        Set a tag for the specified run in async fashion.

        Args:
            run_id: String id for the run
            tag: :py:class:`mlflow.entities.RunTag` instance to set
        """
    return self.log_batch_async(run_id, metrics=[], params=[], tags=[tag])