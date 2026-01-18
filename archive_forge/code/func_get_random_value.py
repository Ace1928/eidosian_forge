from __future__ import annotations
import math
import random
from typing import Any, Callable
from gradio_client.documentation import document
from gradio.components.base import FormComponent
from gradio.events import Events
def get_random_value(self):
    n_steps = int((self.maximum - self.minimum) / self.step)
    step = random.randint(0, n_steps)
    value = self.minimum + step * self.step
    n_decimals = max(str(self.step)[::-1].find('.'), 0)
    if n_decimals:
        value = round(value, n_decimals)
    return value