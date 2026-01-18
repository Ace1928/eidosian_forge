from __future__ import annotations
import base64
import math
import re
import warnings
import httpx
import yaml
from huggingface_hub import InferenceClient
from gradio import components
def encode_to_base64(r: httpx.Response) -> str:
    base64_repr = base64.b64encode(r.content).decode('utf-8')
    data_prefix = ';base64,'
    if data_prefix in base64_repr:
        return base64_repr
    else:
        content_type = r.headers.get('content-type')
        if content_type == 'application/json':
            try:
                data = r.json()[0]
                content_type = data['content-type']
                base64_repr = data['blob']
            except KeyError as ke:
                raise ValueError('Cannot determine content type returned by external API.') from ke
        else:
            pass
        new_base64 = f'data:{content_type};base64,{base64_repr}'
        return new_base64