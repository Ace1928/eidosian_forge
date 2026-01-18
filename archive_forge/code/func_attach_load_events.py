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
def attach_load_events(self):
    """Add a load event for every component whose initial value should be randomized."""
    if Context.root_block:
        for component in Context.root_block.blocks.values():
            if isinstance(component, components.Component) and component.load_event_to_attach:
                load_fn, every = component.load_event_to_attach
                dep = self.set_event_trigger([EventListenerMethod(self, 'load')], load_fn, None, component, no_target=True, queue=False if every is None else None, every=every)[0]
                component.load_event = dep