from __future__ import annotations
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.utils import get_from_dict_or_env
def create_prem_retry_decorator(embedder: PremAIEmbeddings, *, max_retries: int=1) -> Callable[[Any], Any]:
    import premai.models
    errors = [premai.models.api_response_validation_error.APIResponseValidationError, premai.models.conflict_error.ConflictError, premai.models.model_not_found_error.ModelNotFoundError, premai.models.permission_denied_error.PermissionDeniedError, premai.models.provider_api_connection_error.ProviderAPIConnectionError, premai.models.provider_api_status_error.ProviderAPIStatusError, premai.models.provider_api_timeout_error.ProviderAPITimeoutError, premai.models.provider_internal_server_error.ProviderInternalServerError, premai.models.provider_not_found_error.ProviderNotFoundError, premai.models.rate_limit_error.RateLimitError, premai.models.unprocessable_entity_error.UnprocessableEntityError, premai.models.validation_error.ValidationError]
    decorator = create_base_retry_decorator(error_types=errors, max_retries=max_retries, run_manager=None)
    return decorator