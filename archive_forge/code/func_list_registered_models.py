import logging
import os
import shutil
import sys
import time
import urllib
from os.path import join
from typing import List
from mlflow.entities.model_registry import (
from mlflow.entities.model_registry.model_version_stages import (
from mlflow.environment_variables import MLFLOW_REGISTRY_DIR
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
from mlflow.store.artifact.utils.models import _parse_model_uri
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry import (
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.utils.file_utils import (
from mlflow.utils.search_utils import SearchModelUtils, SearchModelVersionUtils, SearchUtils
from mlflow.utils.string_utils import is_string_type
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import (
from mlflow.utils.validation import (
def list_registered_models(self, max_results, page_token):
    """
        List of all registered models.

        Args:
            max_results: Maximum number of registered models desired.
            page_token: Token specifying the next page of results. It should be obtained from
                a ``list_registered_models`` call.

        Returns:
            A PagedList of :py:class:`mlflow.entities.model_registry.RegisteredModel` objects
            that satisfy the search expressions. The pagination token for the next page can be
            obtained via the ``token`` attribute of the object.

        """
    return self.search_registered_models(max_results=max_results, page_token=page_token)