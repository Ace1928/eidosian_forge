import importlib
import importlib.metadata
import os
import shlex
import sys
import textwrap
import types
from flask import Flask, Response, send_from_directory
from packaging.version import Version
from mlflow.exceptions import MlflowException
from mlflow.server import handlers
from mlflow.server.handlers import (
from mlflow.utils.os import get_entry_points, is_windows
from mlflow.utils.process import _exec_cmd
from mlflow.version import VERSION
def get_app_client(app_name: str, *args, **kwargs):
    """
    Instantiate a client provided by an app.

    Args:
        app_name: The app name defined in `setup.py`, e.g., "basic-auth".
        args: Additional arguments passed to the app client constructor.
        kwargs: Additional keyword arguments passed to the app client constructor.

    Returns:
        An app client instance.
    """
    clients = get_entry_points('mlflow.app.client')
    for client in clients:
        if client.name == app_name:
            cls = client.load()
            return cls(*args, **kwargs)
    raise MlflowException(f"Failed to find client for '{app_name}'. Available clients: {[c.name for c in clients]}")