from __future__ import annotations
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.types.llm import (
def postprocess_output_for_llm_inference_task(data: List[str], output_tensors: List[List[int]], pipeline, flavor_config, model_config, inference_task):
    """
    Wrap output data with usage information according to the MLflow inference task.

    Example:
        .. code-block:: python
            data = ["How to learn Python in 3 weeks?"]
            output_tensors = [
                [
                    1128,
                    304,
                    ...,
                    29879,
                ]
            ]
            output_dicts = postprocess_output_for_llm_inference_task(data, output_tensors, **kwargs)

            assert output_dicts == [
                {
                    "id": "e4f3b3e3-3b3e-4b3e-8b3e-3b3e4b3e8b3e",
                    "object": "text_completion",
                    "created": 1707466970,
                    "model": "loaded_model_name",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "length",
                            "text": "1. Start with a beginner's",
                        }
                    ],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 10, "total_tokens": 19},
                }
            ]

    Args:
        data: List of text input prompts.
        output_tensors: List of output tensors that contain the generated tokens (including
            the prompt tokens) corresponding to each input prompt.
        pipeline: The pipeline object used for inference.
        flavor_config: The flavor configuration dictionary for the model.
        model_config: The model configuration dictionary used for inference.
        inference_task: The MLflow inference task.

    Returns:
        List of dictionaries containing the output text and usage information for each input prompt.
    """
    output_dicts = []
    for input_data, output_tensor in zip(data, output_tensors):
        output_dict = _get_output_and_usage_from_tensor(input_data, output_tensor, pipeline, flavor_config, model_config, inference_task)
        output_dicts.append(output_dict)
    return output_dicts