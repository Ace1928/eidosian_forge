from __future__ import annotations
import copy
import hashlib
import inspect
import json
import os
import random
import secrets
import string
import sys
import threading
import time
import warnings
import webbrowser
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Literal, Sequence, cast
from urllib.parse import urlparse, urlunparse
import anyio
import fastapi
import httpx
from anyio import CapacityLimiter
from gradio_client import utils as client_utils
from gradio_client.documentation import document
from gradio import (
from gradio.blocks_events import BlocksEvents, BlocksMeta
from gradio.context import Context
from gradio.data_classes import FileData, GradioModel, GradioRootModel
from gradio.events import (
from gradio.exceptions import (
from gradio.helpers import create_tracker, skip, special_args
from gradio.state_holder import SessionState
from gradio.themes import Default as DefaultTheme
from gradio.themes import ThemeClass as Theme
from gradio.tunneling import (
from gradio.utils import (
def get_block_instance(id: int) -> Block:
    for block_config in components_config:
        if block_config['id'] == id:
            break
    else:
        raise ValueError(f'Cannot find block with id {id}')
    cls = component_or_layout_class(block_config['props']['name'])
    if block_config['props'].get('proxy_url') is None:
        block_config['props']['proxy_url'] = f'{proxy_url}/'
    postprocessed_value = block_config['props'].pop('value', None)
    constructor_args = cls.recover_kwargs(block_config['props'])
    block = cls(**constructor_args)
    if postprocessed_value is not None:
        block.value = postprocessed_value
    block_proxy_url = block_config['props']['proxy_url']
    block.proxy_url = block_proxy_url
    proxy_urls.add(block_proxy_url)
    if (_selectable := block_config['props'].pop('_selectable', None)) is not None:
        block._selectable = _selectable
    return block