import threading
from typing import Any, Dict, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
def standardize_model_name(model_name: str, is_completion: bool=False) -> str:
    """
    Standardize the model name to a format that can be used in the OpenAI API.

    Args:
        model_name: Model name to standardize.
        is_completion: Whether the model is used for completion or not.
            Defaults to False.

    Returns:
        Standardized model name.

    """
    model_name = model_name.lower()
    if '.ft-' in model_name:
        model_name = model_name.split('.ft-')[0] + '-azure-finetuned'
    if ':ft-' in model_name:
        model_name = model_name.split(':')[0] + '-finetuned-legacy'
    if 'ft:' in model_name:
        model_name = model_name.split(':')[1] + '-finetuned'
    if is_completion and (model_name.startswith('gpt-4') or model_name.startswith('gpt-3.5') or model_name.startswith('gpt-35') or ('finetuned' in model_name and 'legacy' not in model_name)):
        return model_name + '-completion'
    else:
        return model_name