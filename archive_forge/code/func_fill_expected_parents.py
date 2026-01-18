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
def fill_expected_parents(self):
    children = []
    pseudo_parent = None
    for child in self.children:
        expected_parent = child.get_expected_parent()
        if not expected_parent or isinstance(self, expected_parent):
            pseudo_parent = None
            children.append(child)
        else:
            if pseudo_parent is not None and isinstance(pseudo_parent, expected_parent):
                pseudo_parent.add_child(child)
            else:
                pseudo_parent = expected_parent(render=False)
                pseudo_parent.parent = self
                children.append(pseudo_parent)
                pseudo_parent.add_child(child)
                if Context.root_block:
                    Context.root_block.blocks[pseudo_parent._id] = pseudo_parent
            child.parent = pseudo_parent
    self.children = children