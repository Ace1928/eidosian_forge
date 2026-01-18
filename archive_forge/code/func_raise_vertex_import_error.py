from importlib import metadata
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
from langchain_core.callbacks import (
from langchain_core.language_models.llms import BaseLLM, create_base_retry_decorator
def raise_vertex_import_error(minimum_expected_version: str='1.38.0') -> None:
    """Raise ImportError related to Vertex SDK being not available.

    Args:
        minimum_expected_version: The lowest expected version of the SDK.
    Raises:
        ImportError: an ImportError that mentions a required version of the SDK.
    """
    raise ImportError(f'Please, install or upgrade the google-cloud-aiplatform library: pip install google-cloud-aiplatform>={minimum_expected_version}')