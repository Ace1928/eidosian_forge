from __future__ import annotations
import math
from copy import deepcopy
from typing import Any, Callable
from gradio_client.documentation import document
from gradio.components.base import Component
@property
def skip_api(self):
    return True