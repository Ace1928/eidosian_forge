import json
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Type, Union
from langchain_core.output_parsers import (
from langchain_core.output_parsers.openai_functions import (
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import (
from langchain.output_parsers import (
def create_openai_fn_runnable(functions: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable]], llm: Runnable, prompt: Optional[BasePromptTemplate]=None, *, enforce_single_function_usage: bool=True, output_parser: Optional[Union[BaseOutputParser, BaseGenerationOutputParser]]=None, **llm_kwargs: Any) -> Runnable:
    """Create a runnable sequence that uses OpenAI functions.

    Args:
        functions: A sequence of either dictionaries, pydantic.BaseModels classes, or
            Python functions. If dictionaries are passed in, they are assumed to
            already be a valid OpenAI functions. If only a single
            function is passed in, then it will be enforced that the model use that
            function. pydantic.BaseModels and Python functions should have docstrings
            describing what the function does. For best results, pydantic.BaseModels
            should have descriptions of the parameters and Python functions should have
            Google Python style args descriptions in the docstring. Additionally,
            Python functions should only use primitive types (str, int, float, bool) or
            pydantic.BaseModels for arguments.
        llm: Language model to use, assumed to support the OpenAI function-calling API.
        prompt: BasePromptTemplate to pass to the model.
        enforce_single_function_usage: only used if a single function is passed in. If
            True, then the model will be forced to use the given function. If False,
            then the model will be given the option to use the given function or not.
        output_parser: BaseLLMOutputParser to use for parsing model outputs. By default
            will be inferred from the function types. If pydantic.BaseModels are passed
            in, then the OutputParser will try to parse outputs using those. Otherwise
            model outputs will simply be parsed as JSON. If multiple functions are
            passed in and they are not pydantic.BaseModels, the chain output will
            include both the name of the function that was returned and the arguments
            to pass to the function.
        **llm_kwargs: Additional named arguments to pass to the language model.

    Returns:
        A runnable sequence that will pass in the given functions to the model when run.

    Example:
        .. code-block:: python

                from typing import Optional

                from langchain.chains.structured_output import create_openai_fn_runnable
                from langchain_openai import ChatOpenAI
                from langchain_core.pydantic_v1 import BaseModel, Field


                class RecordPerson(BaseModel):
                    '''Record some identifying information about a person.'''

                    name: str = Field(..., description="The person's name")
                    age: int = Field(..., description="The person's age")
                    fav_food: Optional[str] = Field(None, description="The person's favorite food")


                class RecordDog(BaseModel):
                    '''Record some identifying information about a dog.'''

                    name: str = Field(..., description="The dog's name")
                    color: str = Field(..., description="The dog's color")
                    fav_food: Optional[str] = Field(None, description="The dog's favorite food")


                llm = ChatOpenAI(model="gpt-4", temperature=0)
                structured_llm = create_openai_fn_runnable([RecordPerson, RecordDog], llm)
                structured_llm.invoke("Harry was a chubby brown beagle who loved chicken)
                # -> RecordDog(name="Harry", color="brown", fav_food="chicken")
    """
    if not functions:
        raise ValueError('Need to pass in at least one function. Received zero.')
    openai_functions = [convert_to_openai_function(f) for f in functions]
    llm_kwargs_: Dict[str, Any] = {'functions': openai_functions, **llm_kwargs}
    if len(openai_functions) == 1 and enforce_single_function_usage:
        llm_kwargs_['function_call'] = {'name': openai_functions[0]['name']}
    output_parser = output_parser or get_openai_output_parser(functions)
    if prompt:
        return prompt | llm.bind(**llm_kwargs_) | output_parser
    else:
        return llm.bind(**llm_kwargs_) | output_parser