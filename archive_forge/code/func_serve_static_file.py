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
@app.route(_add_static_prefix('/static-files/<path:path>'))
def serve_static_file(path):
    if IS_FLASK_V1:
        return send_from_directory(app.static_folder, path, cache_timeout=2419200)
    else:
        return send_from_directory(app.static_folder, path, max_age=2419200)