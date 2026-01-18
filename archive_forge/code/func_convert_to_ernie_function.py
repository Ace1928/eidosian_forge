import inspect
from typing import (
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import (
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable
from langchain.chains import LLMChain
from langchain.output_parsers.ernie_functions import (
from langchain.utils.ernie_functions import convert_pydantic_to_ernie_function
def convert_to_ernie_function(function: Union[Dict[str, Any], Type[BaseModel], Callable]) -> Dict[str, Any]:
    """Convert a raw function/class to an Ernie function.

    Args:
        function: Either a dictionary, a pydantic.BaseModel class, or a Python function.
            If a dictionary is passed in, it is assumed to already be a valid Ernie
            function.

    Returns:
        A dict version of the passed in function which is compatible with the
            Ernie function-calling API.
    """
    if isinstance(function, dict):
        return function
    elif isinstance(function, type) and issubclass(function, BaseModel):
        return cast(Dict, convert_pydantic_to_ernie_function(function))
    elif callable(function):
        return convert_python_function_to_ernie_function(function)
    else:
        raise ValueError(f'Unsupported function type {type(function)}. Functions must be passed in as Dict, pydantic.BaseModel, or Callable.')