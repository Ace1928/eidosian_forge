from __future__ import annotations
import base64
import math
import re
import warnings
import httpx
import yaml
from huggingface_hub import InferenceClient
from gradio import components
def sentence_similarity_wrapper(client: InferenceClient):

    def sentence_similarity_inner(input: str, sentences: str):
        return client.sentence_similarity(input, sentences.split('\n'))
    return sentence_similarity_inner