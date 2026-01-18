from typing import Any, Dict
def get_llm_kwargs(function: dict) -> dict:
    """Returns the kwargs for the LLMChain constructor.

    Args:
        function: The function to use.

    Returns:
        The kwargs for the LLMChain constructor.
    """
    return {'functions': [function], 'function_call': {'name': function['name']}}