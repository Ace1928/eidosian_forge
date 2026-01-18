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
def get_config_file(self):
    config = {'version': routes.VERSION, 'mode': self.mode, 'app_id': self.app_id, 'dev_mode': self.dev_mode, 'analytics_enabled': self.analytics_enabled, 'components': [], 'css': self.css, 'js': self.js, 'head': self.head, 'title': self.title or 'Gradio', 'space_id': self.space_id, 'enable_queue': True, 'show_error': getattr(self, 'show_error', False), 'show_api': self.show_api, 'is_colab': utils.colab_check(), 'stylesheets': self.stylesheets, 'theme': self.theme.name, 'protocol': 'sse_v3', 'body_css': {'body_background_fill': self.theme._get_computed_value('body_background_fill'), 'body_text_color': self.theme._get_computed_value('body_text_color'), 'body_background_fill_dark': self.theme._get_computed_value('body_background_fill_dark'), 'body_text_color_dark': self.theme._get_computed_value('body_text_color_dark')}, 'fill_height': self.fill_height}

    def get_layout(block):
        if not isinstance(block, BlockContext):
            return {'id': block._id}
        children_layout = []
        for child in block.children:
            children_layout.append(get_layout(child))
        return {'id': block._id, 'children': children_layout}
    config['layout'] = get_layout(self)
    for _id, block in self.blocks.items():
        props = block.get_config() if hasattr(block, 'get_config') else {}
        block_config = {'id': _id, 'type': block.get_block_name(), 'props': utils.delete_none(props)}
        block_config['skip_api'] = block.skip_api
        block_config['component_class_id'] = getattr(block, 'component_class_id', None)
        if not block.skip_api:
            block_config['api_info'] = block.api_info()
            block_config['example_inputs'] = block.example_inputs()
        config['components'].append(block_config)
    config['dependencies'] = self.dependencies
    return config