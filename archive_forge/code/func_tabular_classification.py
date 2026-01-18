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
def tabular_classification(self, table: Dict[str, Any], *, model: Optional[str]=None) -> List[str]:
    """
        Classifying a target category (a group) based on a set of attributes.

        Args:
            table (`Dict[str, Any]`):
                Set of attributes to classify.
            model (`str`, *optional*):
                The model to use for the tabular classification task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended tabular classification model will be used.
                Defaults to None.

        Returns:
            `List`: a list of labels, one per row in the initial table.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> table = {
        ...     "fixed_acidity": ["7.4", "7.8", "10.3"],
        ...     "volatile_acidity": ["0.7", "0.88", "0.32"],
        ...     "citric_acid": ["0", "0", "0.45"],
        ...     "residual_sugar": ["1.9", "2.6", "6.4"],
        ...     "chlorides": ["0.076", "0.098", "0.073"],
        ...     "free_sulfur_dioxide": ["11", "25", "5"],
        ...     "total_sulfur_dioxide": ["34", "67", "13"],
        ...     "density": ["0.9978", "0.9968", "0.9976"],
        ...     "pH": ["3.51", "3.2", "3.23"],
        ...     "sulphates": ["0.56", "0.68", "0.82"],
        ...     "alcohol": ["9.4", "9.8", "12.6"],
        ... }
        >>> client.tabular_classification(table=table, model="julien-c/wine-quality")
        ["5", "5", "5"]
        ```
        """
    response = self.post(json={'table': table}, model=model, task='tabular-classification')
    return _bytes_to_list(response)