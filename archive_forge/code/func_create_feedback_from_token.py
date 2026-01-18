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
def create_feedback_from_token(self, token_or_url: Union[str, uuid.UUID], score: Union[float, int, bool, None]=None, *, value: Union[float, int, bool, str, dict, None]=None, correction: Union[dict, None]=None, comment: Union[str, None]=None, metadata: Optional[dict]=None) -> None:
    """Create feedback from a presigned token or URL.

        Args:
            token_or_url (Union[str, uuid.UUID]): The token or URL from which to create
                 feedback.
            score (Union[float, int, bool, None], optional): The score of the feedback.
                Defaults to None.
            value (Union[float, int, bool, str, dict, None], optional): The value of the
                feedback. Defaults to None.
            correction (Union[dict, None], optional): The correction of the feedback.
                Defaults to None.
            comment (Union[str, None], optional): The comment of the feedback. Defaults
                to None.
            metadata (Optional[dict], optional): Additional metadata for the feedback.
                Defaults to None.

        Raises:
            ValueError: If the source API URL is invalid.

        Returns:
            None: This method does not return anything.
        """
    source_api_url, token_uuid = _parse_token_or_url(token_or_url, self.api_url, num_parts=1)
    if source_api_url != self.api_url:
        raise ValueError(f'Invalid source API URL. {source_api_url}')
    response = self.session.post(f'{source_api_url}/feedback/tokens/{_as_uuid(token_uuid)}', data=_dumps_json({'score': score, 'value': value, 'correction': correction, 'comment': comment, 'metadata': metadata}), headers=self._headers)
    ls_utils.raise_for_status_with_text(response)