from __future__ import annotations
import concurrent.futures
import hashlib
import json
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
import urllib.parse
import uuid
import warnings
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Literal
import httpx
import huggingface_hub
from huggingface_hub import CommitOperationAdd, SpaceHardware, SpaceStage
from huggingface_hub.utils import (
from packaging import version
from gradio_client import utils
from gradio_client.compatibility import EndpointV3Compatibility
from gradio_client.data_classes import ParameterInfo
from gradio_client.documentation import document
from gradio_client.exceptions import AuthenticationError
from gradio_client.utils import (
def make_predict(self, helper: Communicator | None=None):

    def _predict(*data) -> tuple:
        data = {'data': data, 'fn_index': self.fn_index, 'session_hash': self.client.session_hash}
        hash_data = {'fn_index': self.fn_index, 'session_hash': self.client.session_hash}
        if self.protocol == 'sse':
            result = utils.synchronize_async(self._sse_fn_v0, data, hash_data, helper)
        elif self.protocol in ('sse_v1', 'sse_v2', 'sse_v2.1', 'sse_v3'):
            event_id = utils.synchronize_async(self.client.send_data, data, hash_data, self.protocol)
            self.client.pending_event_ids.add(event_id)
            self.client.pending_messages_per_event[event_id] = []
            result = utils.synchronize_async(self._sse_fn_v1plus, helper, event_id, self.protocol)
        else:
            raise ValueError(f'Unsupported protocol: {self.protocol}')
        if 'error' in result:
            raise ValueError(result['error'])
        try:
            output = result['data']
        except KeyError as ke:
            is_public_space = self.client.space_id and (not huggingface_hub.space_info(self.client.space_id).private)
            if 'error' in result and '429' in result['error'] and is_public_space:
                raise utils.TooManyRequestsError(f'Too many requests to the API, please try again later. To avoid being rate-limited, please duplicate the Space using Client.duplicate({self.client.space_id}) and pass in your Hugging Face token.') from None
            elif 'error' in result:
                raise ValueError(result['error']) from None
            raise KeyError(f"Could not find 'data' key in response. Response received: {result}") from ke
        return tuple(output)
    return _predict