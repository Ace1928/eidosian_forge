from __future__ import annotations
import ast
import csv
import inspect
import os
import shutil
import subprocess
import tempfile
import warnings
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Optional
import numpy as np
import PIL
import PIL.Image
from gradio_client import utils as client_utils
from gradio_client.documentation import document
from gradio import components, oauth, processing_utils, routes, utils, wasm_utils
from gradio.context import Context, LocalContext
from gradio.data_classes import GradioModel, GradioRootModel
from gradio.events import EventData
from gradio.exceptions import Error
from gradio.flagging import CSVLogger
def special_args(fn: Callable, inputs: list[Any] | None=None, request: routes.Request | None=None, event_data: EventData | None=None) -> tuple[list, int | None, int | None]:
    """
    Checks if function has special arguments Request or EventData (via annotation) or Progress (via default value).
    If inputs is provided, these values will be loaded into the inputs array.
    Parameters:
        fn: function to check.
        inputs: array to load special arguments into.
        request: request to load into inputs.
        event_data: event-related data to load into inputs.
    Returns:
        updated inputs, progress index, event data index.
    """
    try:
        signature = inspect.signature(fn)
    except ValueError:
        return (inputs or [], None, None)
    type_hints = utils.get_type_hints(fn)
    positional_args = []
    for param in signature.parameters.values():
        if param.kind not in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            break
        positional_args.append(param)
    progress_index = None
    event_data_index = None
    for i, param in enumerate(positional_args):
        type_hint = type_hints.get(param.name)
        if isinstance(param.default, Progress):
            progress_index = i
            if inputs is not None:
                inputs.insert(i, param.default)
        elif type_hint == routes.Request:
            if inputs is not None:
                inputs.insert(i, request)
        elif type_hint in (Optional[oauth.OAuthProfile], Optional[oauth.OAuthToken], oauth.OAuthProfile, oauth.OAuthToken):
            if inputs is not None:
                session = getattr(request, 'session', {}) or getattr(getattr(request, 'request', None), 'session', {})
                if type_hint in (Optional[oauth.OAuthProfile], oauth.OAuthProfile):
                    oauth_profile = session['oauth_info']['userinfo'] if 'oauth_info' in session else None
                    if oauth_profile is not None:
                        oauth_profile = oauth.OAuthProfile(oauth_profile)
                    elif type_hint == oauth.OAuthProfile:
                        raise Error('This action requires a logged in user. Please sign in and retry.')
                    inputs.insert(i, oauth_profile)
                elif type_hint in (Optional[oauth.OAuthToken], oauth.OAuthToken):
                    oauth_info = session.get('oauth_info', None)
                    oauth_token = oauth.OAuthToken(token=oauth_info['access_token'], scope=oauth_info['scope'], expires_at=oauth_info['expires_at']) if oauth_info is not None else None
                    if oauth_token is None and type_hint == oauth.OAuthToken:
                        raise Error('This action requires a logged in user. Please sign in and retry.')
                    inputs.insert(i, oauth_token)
        elif type_hint and inspect.isclass(type_hint) and issubclass(type_hint, EventData):
            event_data_index = i
            if inputs is not None and event_data is not None:
                inputs.insert(i, type_hint(event_data.target, event_data._data))
        elif param.default is not param.empty and inputs is not None and (len(inputs) <= i):
            inputs.insert(i, param.default)
    if inputs is not None:
        while len(inputs) < len(positional_args):
            i = len(inputs)
            param = positional_args[i]
            if param.default == param.empty:
                warnings.warn('Unexpected argument. Filling with None.')
                inputs.append(None)
            else:
                inputs.append(param.default)
    return (inputs or [], progress_index, event_data_index)