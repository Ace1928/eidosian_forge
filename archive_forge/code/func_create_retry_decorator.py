from importlib import metadata
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
from langchain_core.callbacks import (
from langchain_core.language_models.llms import BaseLLM, create_base_retry_decorator
def create_retry_decorator(llm: BaseLLM, *, max_retries: int=1, run_manager: Optional[Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]]=None) -> Callable[[Any], Any]:
    """Create a retry decorator for Vertex / Palm LLMs."""
    import google.api_core
    errors = [google.api_core.exceptions.ResourceExhausted, google.api_core.exceptions.ServiceUnavailable, google.api_core.exceptions.Aborted, google.api_core.exceptions.DeadlineExceeded, google.api_core.exceptions.GoogleAPIError]
    decorator = create_base_retry_decorator(error_types=errors, max_retries=max_retries, run_manager=run_manager)
    return decorator