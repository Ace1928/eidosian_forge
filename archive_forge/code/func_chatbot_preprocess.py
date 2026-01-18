from __future__ import annotations
import base64
import math
import re
import warnings
import httpx
import yaml
from huggingface_hub import InferenceClient
from gradio import components
def chatbot_preprocess(text, state):
    if not state:
        return (text, [], [])
    return (text, state['conversation']['generated_responses'], state['conversation']['past_user_inputs'])