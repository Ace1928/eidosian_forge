import atexit
import codecs
import errno
import fnmatch
import gzip
import json
import logging
import math
import os
import pathlib
import posixpath
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from subprocess import CalledProcessError, TimeoutExpired
from typing import Optional, Union
from urllib.parse import unquote
from urllib.request import pathname2url
import yaml
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.entities import FileInfo
from mlflow.environment_variables import (
from mlflow.exceptions import MissingConfigException, MlflowException
from mlflow.protos.databricks_artifacts_pb2 import ArtifactCredentialType
from mlflow.utils import download_cloud_file_chunk, merge_dicts
from mlflow.utils.databricks_utils import _get_dbutils
from mlflow.utils.os import is_windows
from mlflow.utils.process import cache_return_value_per_process
from mlflow.utils.request_utils import cloud_storage_http_request, download_chunk
from mlflow.utils.rest_utils import augmented_raise_for_status
def render_and_merge_yaml(root, template_name, context_name):
    """Renders a Jinja2-templated YAML file based on a YAML context file, merge them, and return
    result as a dictionary.

    Args:
        root: Root directory of the YAML files.
        template_name: Name of the template file.
        context_name: Name of the context file.

    Returns:
        Data in yaml file as dictionary.
    """
    from jinja2 import FileSystemLoader, StrictUndefined
    from jinja2.sandbox import SandboxedEnvironment
    template_path = os.path.join(root, template_name)
    context_path = os.path.join(root, context_name)
    for path in (template_path, context_path):
        if not pathlib.Path(path).is_file():
            raise MissingConfigException(f"Yaml file '{path}' does not exist.")
    j2_env = SandboxedEnvironment(loader=FileSystemLoader(root, encoding=ENCODING), undefined=StrictUndefined, line_comment_prefix='#')

    def from_json(input_var):
        with open(input_var, encoding='utf-8') as f:
            return json.load(f)
    j2_env.filters['from_json'] = from_json
    context_source = j2_env.get_template(context_name).render({})
    context_dict = yaml.load(context_source, Loader=UniqueKeyLoader) or {}
    source = j2_env.get_template(template_name).render(context_dict)
    rendered_template_dict = yaml.load(source, Loader=UniqueKeyLoader)
    return merge_dicts(rendered_template_dict, context_dict)