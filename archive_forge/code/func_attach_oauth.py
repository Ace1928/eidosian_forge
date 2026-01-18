from __future__ import annotations
import hashlib
import os
import typing
import urllib.parse
import warnings
from dataclasses import dataclass, field
import fastapi
from fastapi.responses import RedirectResponse
from huggingface_hub import HfFolder, whoami
from .utils import get_space
def attach_oauth(app: fastapi.FastAPI):
    try:
        from starlette.middleware.sessions import SessionMiddleware
    except ImportError as e:
        raise ImportError('Cannot initialize OAuth to due a missing library. Please run `pip install gradio[oauth]` or add `gradio[oauth]` to your requirements.txt file in order to install the required dependencies.') from e
    if get_space() is not None:
        _add_oauth_routes(app)
    else:
        _add_mocked_oauth_routes(app)
    session_secret = (OAUTH_CLIENT_SECRET or '') + '-v4'
    app.add_middleware(SessionMiddleware, secret_key=hashlib.sha256(session_secret.encode()).hexdigest(), same_site='none', https_only=True)