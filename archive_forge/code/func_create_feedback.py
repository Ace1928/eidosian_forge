from __future__ import annotations
import collections
import datetime
import functools
import importlib
import importlib.metadata
import io
import json
import logging
import os
import random
import re
import socket
import sys
import threading
import time
import uuid
import warnings
import weakref
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue, Queue
from typing import (
from urllib import parse as urllib_parse
import orjson
import requests
from requests import adapters as requests_adapters
from urllib3.util import Retry
import langsmith
from langsmith import env as ls_env
from langsmith import schemas as ls_schemas
from langsmith import utils as ls_utils
def create_feedback(self, run_id: Optional[ID_TYPE], key: str, *, score: Union[float, int, bool, None]=None, value: Union[float, int, bool, str, dict, None]=None, correction: Union[dict, None]=None, comment: Union[str, None]=None, source_info: Optional[Dict[str, Any]]=None, feedback_source_type: Union[ls_schemas.FeedbackSourceType, str]=ls_schemas.FeedbackSourceType.API, source_run_id: Optional[ID_TYPE]=None, feedback_id: Optional[ID_TYPE]=None, feedback_config: Optional[ls_schemas.FeedbackConfig]=None, stop_after_attempt: int=10, project_id: Optional[ID_TYPE]=None, **kwargs: Any) -> ls_schemas.Feedback:
    """Create a feedback in the LangSmith API.

        Parameters
        ----------
        run_id : str or UUID
            The ID of the run to provide feedback for. Either the run_id OR
            the project_id must be provided.
        key : str
            The name of the metric or 'aspect' this feedback is about.
        score : float or int or bool or None, default=None
            The score to rate this run on the metric or aspect.
        value : float or int or bool or str or dict or None, default=None
            The display value or non-numeric value for this feedback.
        correction : dict or None, default=None
            The proper ground truth for this run.
        comment : str or None, default=None
            A comment about this feedback, such as a justification for the score or
            chain-of-thought trajectory for an LLM judge.
        source_info : Dict[str, Any] or None, default=None
            Information about the source of this feedback.
        feedback_source_type : FeedbackSourceType or str, default=FeedbackSourceType.API
            The type of feedback source, such as model (for model-generated feedback)
                or API.
        source_run_id : str or UUID or None, default=None,
            The ID of the run that generated this feedback, if a "model" type.
        feedback_id : str or UUID or None, default=None
            The ID of the feedback to create. If not provided, a random UUID will be
            generated.
        feedback_config: FeedbackConfig or None, default=None,
            The configuration specifying how to interpret feedback with this key.
            Examples include continuous (with min/max bounds), categorical,
            or freeform.
        stop_after_attempt : int, default=10
            The number of times to retry the request before giving up.
        project_id : str or UUID
            The ID of the project_id to provide feedback on. One - and only one - of
            this and run_id must be provided.
        """
    if run_id is None and project_id is None:
        raise ValueError('One of run_id and project_id must be provided')
    if run_id is not None and project_id is not None:
        raise ValueError('Only one of run_id and project_id must be provided')
    if kwargs:
        warnings.warn(f'The following arguments are no longer used in the create_feedback endpoint: {sorted(kwargs)}', DeprecationWarning)
    if not isinstance(feedback_source_type, ls_schemas.FeedbackSourceType):
        feedback_source_type = ls_schemas.FeedbackSourceType(feedback_source_type)
    if feedback_source_type == ls_schemas.FeedbackSourceType.API:
        feedback_source: ls_schemas.FeedbackSourceBase = ls_schemas.APIFeedbackSource(metadata=source_info)
    elif feedback_source_type == ls_schemas.FeedbackSourceType.MODEL:
        feedback_source = ls_schemas.ModelFeedbackSource(metadata=source_info)
    else:
        raise ValueError(f'Unknown feedback source type {feedback_source_type}')
    feedback_source.metadata = feedback_source.metadata if feedback_source.metadata is not None else {}
    if source_run_id is not None and '__run' not in feedback_source.metadata:
        feedback_source.metadata['__run'] = {'run_id': str(source_run_id)}
    if feedback_source.metadata and '__run' in feedback_source.metadata:
        _run_meta: Union[dict, Any] = feedback_source.metadata['__run']
        if hasattr(_run_meta, 'dict') and callable(_run_meta):
            _run_meta = _run_meta.dict()
        if 'run_id' in _run_meta:
            _run_meta['run_id'] = str(_as_uuid(feedback_source.metadata['__run']['run_id'], "feedback_source.metadata['__run']['run_id']"))
        feedback_source.metadata['__run'] = _run_meta
    feedback = ls_schemas.FeedbackCreate(id=feedback_id or uuid.uuid4(), run_id=run_id, key=key, score=score, value=value, correction=correction, comment=comment, feedback_source=feedback_source, created_at=datetime.datetime.now(datetime.timezone.utc), modified_at=datetime.datetime.now(datetime.timezone.utc), feedback_config=feedback_config, session_id=project_id)
    self.request_with_retries('POST', '/feedback', request_kwargs={'data': _dumps_json(feedback.dict(exclude_none=True))}, stop_after_attempt=stop_after_attempt, retry_on=(ls_utils.LangSmithNotFoundError,))
    return ls_schemas.Feedback(**feedback.dict())