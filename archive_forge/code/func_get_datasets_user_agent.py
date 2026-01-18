import copy
import io
import json
import multiprocessing
import os
import posixpath
import re
import shutil
import sys
import time
import urllib
import warnings
from contextlib import closing, contextmanager
from functools import partial
from pathlib import Path
from typing import Optional, TypeVar, Union
from unittest.mock import patch
from urllib.parse import urljoin, urlparse
import fsspec
import huggingface_hub
import requests
from fsspec.core import strip_protocol
from fsspec.utils import can_be_local
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from .. import __version__, config
from ..download.download_config import DownloadConfig
from . import _tqdm, logging
from . import tqdm as hf_tqdm
from ._filelock import FileLock
from .extract import ExtractManager
def get_datasets_user_agent(user_agent: Optional[Union[str, dict]]=None) -> str:
    ua = f'datasets/{__version__}'
    ua += f'; python/{config.PY_VERSION}'
    ua += f'; huggingface_hub/{huggingface_hub.__version__}'
    ua += f'; pyarrow/{config.PYARROW_VERSION}'
    if config.TORCH_AVAILABLE:
        ua += f'; torch/{config.TORCH_VERSION}'
    if config.TF_AVAILABLE:
        ua += f'; tensorflow/{config.TF_VERSION}'
    if config.JAX_AVAILABLE:
        ua += f'; jax/{config.JAX_VERSION}'
    if config.BEAM_AVAILABLE:
        ua += f'; apache_beam/{config.BEAM_VERSION}'
    if isinstance(user_agent, dict):
        ua += f'; {'; '.join((f'{k}/{v}' for k, v in user_agent.items()))}'
    elif isinstance(user_agent, str):
        ua += '; ' + user_agent
    return ua