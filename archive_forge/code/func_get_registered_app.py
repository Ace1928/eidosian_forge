from __future__ import annotations
import sys
from contextlib import contextmanager
from contextvars import ContextVar
def get_registered_app(app_id: str):
    try:
        return app_map[app_id]
    except KeyError as e:
        raise GradioAppNotFoundError(f'Gradio app not found (ID: {app_id}). Forgot to call demo.launch()?') from e