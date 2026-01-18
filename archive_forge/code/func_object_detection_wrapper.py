from __future__ import annotations
import base64
import math
import re
import warnings
import httpx
import yaml
from huggingface_hub import InferenceClient
from gradio import components
def object_detection_wrapper(client: InferenceClient):

    def object_detection_inner(input: str):
        annotations = client.object_detection(input)
        formatted_annotations = [((a['box']['xmin'], a['box']['ymin'], a['box']['xmax'], a['box']['ymax']), a['label']) for a in annotations]
        return (input, formatted_annotations)
    return object_detection_inner