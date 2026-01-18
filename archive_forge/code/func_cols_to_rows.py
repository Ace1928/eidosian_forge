from __future__ import annotations
import base64
import math
import re
import warnings
import httpx
import yaml
from huggingface_hub import InferenceClient
from gradio import components
def cols_to_rows(example_data: dict[str, list[float]]) -> tuple[list[str], list[list[float]]]:
    headers = list(example_data.keys())
    n_rows = max((len(example_data[header] or []) for header in headers))
    data = []
    for row_index in range(n_rows):
        row_data = []
        for header in headers:
            col = example_data[header] or []
            if row_index >= len(col):
                row_data.append('NaN')
            else:
                row_data.append(col[row_index])
        data.append(row_data)
    return (headers, data)