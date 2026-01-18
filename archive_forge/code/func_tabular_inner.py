from __future__ import annotations
import base64
import math
import re
import warnings
import httpx
import yaml
from huggingface_hub import InferenceClient
from gradio import components
def tabular_inner(data):
    if pipeline not in ('tabular_classification', 'tabular_regression'):
        raise TypeError(f'pipeline type {pipeline!r} not supported')
    assert client.model
    if pipeline == 'tabular_classification':
        return client.tabular_classification(data, model=client.model)
    else:
        return client.tabular_regression(data, model=client.model)