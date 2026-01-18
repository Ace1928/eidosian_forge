import threading
from typing import Any, Dict, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
def get_openai_token_cost_for_model(model_name: str, num_tokens: int, is_completion: bool=False) -> float:
    """
    Get the cost in USD for a given model and number of tokens.

    Args:
        model_name: Name of the model
        num_tokens: Number of tokens.
        is_completion: Whether the model is used for completion or not.
            Defaults to False.

    Returns:
        Cost in USD.
    """
    model_name = standardize_model_name(model_name, is_completion=is_completion)
    if model_name not in MODEL_COST_PER_1K_TOKENS:
        raise ValueError(f'Unknown model: {model_name}. Please provide a valid OpenAI model name.Known models are: ' + ', '.join(MODEL_COST_PER_1K_TOKENS.keys()))
    return MODEL_COST_PER_1K_TOKENS[model_name] * (num_tokens / 1000)