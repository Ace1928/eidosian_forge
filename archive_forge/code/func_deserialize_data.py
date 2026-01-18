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
def deserialize_data(self, fn_index: int, outputs: list[Any]) -> list[Any]:
    dependency = self.dependencies[fn_index]
    predictions = []
    for o, output_id in enumerate(dependency['outputs']):
        try:
            block = self.blocks[output_id]
        except KeyError as e:
            raise InvalidBlockError(f'Output component with id {output_id} used in {dependency['trigger']}() event not found in this gr.Blocks context. You are allowed to nest gr.Blocks contexts, but there must be a gr.Blocks context that contains all components and events.') from e
        if not isinstance(block, components.Component):
            raise InvalidComponentError(f'{block.__class__} Component with id {output_id} not a valid output component.')
        deserialized = client_utils.traverse(outputs[o], lambda s: s['path'], client_utils.is_file_obj)
        predictions.append(deserialized)
    return predictions