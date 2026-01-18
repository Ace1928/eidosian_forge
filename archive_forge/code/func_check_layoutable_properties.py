import asyncio
import contextlib
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from queue import Empty, Queue
from threading import Thread
import numpy as np
import pytest
import requests
from packaging.version import Version
import panel as pn
from panel.io.server import serve
from panel.io.state import state
from panel.pane.alert import Alert
from panel.pane.markup import Markdown
from panel.widgets.button import _ButtonBase
def check_layoutable_properties(layoutable, model):
    layoutable.styles = {'background': '#fffff0'}
    assert model.styles['background'] == '#fffff0'
    layoutable.css_classes = ['custom_class']
    if isinstance(layoutable, Alert):
        assert model.css_classes == ['markdown', 'custom_class', 'alert', 'alert-primary']
    elif isinstance(layoutable, Markdown):
        assert model.css_classes == ['markdown', 'custom_class']
    elif isinstance(layoutable, _ButtonBase):
        assert model.css_classes == ['solid', 'custom_class']
    else:
        assert model.css_classes == ['custom_class']
    layoutable.width = 500
    assert model.width == 500
    layoutable.height = 450
    assert model.height == 450
    layoutable.min_height = 300
    assert model.min_height == 300
    layoutable.min_width = 250
    assert model.min_width == 250
    layoutable.max_height = 600
    assert model.max_height == 600
    layoutable.max_width = 550
    assert model.max_width == 550
    layoutable.margin = 10
    assert model.margin == 10
    layoutable.sizing_mode = 'stretch_width'
    assert model.sizing_mode == 'stretch_width'
    layoutable.width_policy = 'max'
    assert model.width_policy == 'max'
    layoutable.height_policy = 'min'
    assert model.height_policy == 'min'