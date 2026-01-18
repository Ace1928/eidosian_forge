from __future__ import annotations
import base64
import math
import re
import warnings
import httpx
import yaml
from huggingface_hub import InferenceClient
from gradio import components
def chatbot_postprocess(response):
    chatbot_history = list(zip(response['conversation']['past_user_inputs'], response['conversation']['generated_responses']))
    return (chatbot_history, response)