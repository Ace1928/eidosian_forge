from __future__ import annotations
import base64
import math
import re
import warnings
import httpx
import yaml
from huggingface_hub import InferenceClient
from gradio import components
def postprocess_mask_tokens(scores: list[dict[str, str | float]]) -> dict:
    return {c['token_str']: c['score'] for c in scores}