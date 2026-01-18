import base64
import logging
import time
import warnings
from dataclasses import asdict
from typing import (
from requests import HTTPError
from requests.structures import CaseInsensitiveDict
from huggingface_hub.constants import ALL_INFERENCE_API_FRAMEWORKS, INFERENCE_ENDPOINT, MAIN_INFERENCE_API_FRAMEWORKS
from huggingface_hub.inference._common import (
from huggingface_hub.inference._text_generation import (
from huggingface_hub.inference._types import (
from huggingface_hub.utils import (
def summarization(self, text: str, *, parameters: Optional[Dict[str, Any]]=None, model: Optional[str]=None) -> str:
    """
        Generate a summary of a given text using a specified model.

        Args:
            text (`str`):
                The input text to summarize.
            parameters (`Dict[str, Any]`, *optional*):
                Additional parameters for summarization. Check out this [page](https://huggingface.co/docs/api-inference/detailed_parameters#summarization-task)
                for more details.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.

        Returns:
            `str`: The generated summary text.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.summarization("The Eiffel tower...")
        'The Eiffel tower is one of the most famous landmarks in the world....'
        ```
        """
    payload: Dict[str, Any] = {'inputs': text}
    if parameters is not None:
        payload['parameters'] = parameters
    response = self.post(json=payload, model=model, task='summarization')
    return _bytes_to_dict(response)[0]['summary_text']