import json
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Type, Union
from langchain_core.output_parsers import (
from langchain_core.output_parsers.openai_functions import (
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import (
from langchain.output_parsers import (
def create_structured_output_runnable(output_schema: Union[Dict[str, Any], Type[BaseModel]], llm: Runnable, prompt: Optional[BasePromptTemplate]=None, *, output_parser: Optional[Union[BaseOutputParser, BaseGenerationOutputParser]]=None, enforce_function_usage: bool=True, return_single: bool=True, mode: Literal['openai-functions', 'openai-tools', 'openai-json']='openai-functions', **kwargs: Any) -> Runnable:
    """Create a runnable for extracting structured outputs.

    Args:
        output_schema: Either a dictionary or pydantic.BaseModel class. If a dictionary
            is passed in, it's assumed to already be a valid JsonSchema.
            For best results, pydantic.BaseModels should have docstrings describing what
            the schema represents and descriptions for the parameters.
        llm: Language model to use. Assumed to support the OpenAI function-calling API 
            if mode is 'openai-function'. Assumed to support OpenAI response_format 
            parameter if mode is 'openai-json'.
        prompt: BasePromptTemplate to pass to the model. If mode is 'openai-json' and 
            prompt has input variable 'output_schema' then the given output_schema 
            will be converted to a JsonSchema and inserted in the prompt.
        output_parser: Output parser to use for parsing model outputs. By default
            will be inferred from the function types. If pydantic.BaseModel is passed
            in, then the OutputParser will try to parse outputs using the pydantic 
            class. Otherwise model outputs will be parsed as JSON.
        mode: How structured outputs are extracted from the model. If 'openai-functions' 
            then OpenAI function calling is used with the deprecated 'functions', 
            'function_call' schema. If 'openai-tools' then OpenAI function 
            calling with the latest 'tools', 'tool_choice' schema is used. This is 
            recommended over 'openai-functions'. If 'openai-json' then OpenAI model 
            with response_format set to JSON is used.
        enforce_function_usage: Only applies when mode is 'openai-tools' or 
            'openai-functions'. If True, then the model will be forced to use the given 
            output schema. If False, then the model can elect whether to use the output 
            schema.
        return_single: Only applies when mode is 'openai-tools'. Whether to a list of 
            structured outputs or a single one. If True and model does not return any 
            structured outputs then chain output is None. If False and model does not 
            return any structured outputs then chain output is an empty list.
        **kwargs: Additional named arguments.

    Returns:
        A runnable sequence that will return a structured output(s) matching the given 
            output_schema.
    
    OpenAI tools example with Pydantic schema (mode='openai-tools'):
        .. code-block:: python
        
                from typing import Optional

                from langchain.chains import create_structured_output_runnable
                from langchain_openai import ChatOpenAI
                from langchain_core.pydantic_v1 import BaseModel, Field


                class RecordDog(BaseModel):
                    '''Record some identifying information about a dog.'''

                    name: str = Field(..., description="The dog's name")
                    color: str = Field(..., description="The dog's color")
                    fav_food: Optional[str] = Field(None, description="The dog's favorite food")

                llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", "You are an extraction algorithm. Please extract every possible instance"), 
                        ('human', '{input}')
                    ]
                )
                structured_llm = create_structured_output_runnable(
                    RecordDog, 
                    llm, 
                    mode="openai-tools", 
                    enforce_function_usage=True, 
                    return_single=True
                )
                structured_llm.invoke({"input": "Harry was a chubby brown beagle who loved chicken"})
                # -> RecordDog(name="Harry", color="brown", fav_food="chicken")
                
    OpenAI tools example with dict schema (mode="openai-tools"):
        .. code-block:: python
        
                from typing import Optional

                from langchain.chains import create_structured_output_runnable
                from langchain_openai import ChatOpenAI


                dog_schema = {
                    "type": "function",
                    "function": {
                        "name": "record_dog",
                        "description": "Record some identifying information about a dog.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "description": "The dog's name",
                                    "type": "string"
                                },
                                "color": {
                                    "description": "The dog's color",
                                    "type": "string"
                                },
                                "fav_food": {
                                    "description": "The dog's favorite food",
                                    "type": "string"
                                }
                            },
                            "required": ["name", "color"]
                        }
                    }
                }


                llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
                structured_llm = create_structured_output_runnable(
                    doc_schema, 
                    llm, 
                    mode="openai-tools", 
                    enforce_function_usage=True, 
                    return_single=True
                )
                structured_llm.invoke("Harry was a chubby brown beagle who loved chicken")
                # -> {'name': 'Harry', 'color': 'brown', 'fav_food': 'chicken'}
    
    OpenAI functions example (mode="openai-functions"):
        .. code-block:: python

                from typing import Optional

                from langchain.chains import create_structured_output_runnable
                from langchain_openai import ChatOpenAI
                from langchain_core.pydantic_v1 import BaseModel, Field

                class Dog(BaseModel):
                    '''Identifying information about a dog.'''

                    name: str = Field(..., description="The dog's name")
                    color: str = Field(..., description="The dog's color")
                    fav_food: Optional[str] = Field(None, description="The dog's favorite food")

                llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
                structured_llm = create_structured_output_runnable(Dog, llm, mode="openai-functions")
                structured_llm.invoke("Harry was a chubby brown beagle who loved chicken")
                # -> Dog(name="Harry", color="brown", fav_food="chicken")
                
    OpenAI functions with prompt example:
        .. code-block:: python

                from typing import Optional

                from langchain.chains import create_structured_output_runnable
                from langchain_openai import ChatOpenAI
                from langchain_core.prompts import ChatPromptTemplate
                from langchain_core.pydantic_v1 import BaseModel, Field

                class Dog(BaseModel):
                    '''Identifying information about a dog.'''

                    name: str = Field(..., description="The dog's name")
                    color: str = Field(..., description="The dog's color")
                    fav_food: Optional[str] = Field(None, description="The dog's favorite food")

                llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
                structured_llm = create_structured_output_runnable(Dog, llm, mode="openai-functions")
                system = '''Extract information about any dogs mentioned in the user input.'''
                prompt = ChatPromptTemplate.from_messages(
                    [("system", system), ("human", "{input}"),]
                )
                chain = prompt | structured_llm
                chain.invoke({"input": "Harry was a chubby brown beagle who loved chicken"})
                # -> Dog(name="Harry", color="brown", fav_food="chicken")
    OpenAI json response format example (mode="openai-json"):
        .. code-block:: python
        
                from typing import Optional

                from langchain.chains import create_structured_output_runnable
                from langchain_openai import ChatOpenAI
                from langchain_core.prompts import ChatPromptTemplate
                from langchain_core.pydantic_v1 import BaseModel, Field

                class Dog(BaseModel):
                    '''Identifying information about a dog.'''

                    name: str = Field(..., description="The dog's name")
                    color: str = Field(..., description="The dog's color")
                    fav_food: Optional[str] = Field(None, description="The dog's favorite food")

                llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
                structured_llm = create_structured_output_runnable(Dog, llm, mode="openai-json")
                system = '''You are a world class assistant for extracting information in structured JSON formats.                 
                Extract a valid JSON blob from the user input that matches the following JSON Schema:
                
                {output_schema}'''
                prompt = ChatPromptTemplate.from_messages(
                    [("system", system), ("human", "{input}"),]
                )
                chain = prompt | structured_llm
                chain.invoke({"input": "Harry was a chubby brown beagle who loved chicken"})
    """
    force_function_usage = kwargs.get('enforce_single_function_usage', enforce_function_usage)
    if mode == 'openai-tools':
        keys_in_kwargs = set(kwargs.keys())
        unrecognized_keys = keys_in_kwargs - {'enforce_single_function_usage'}
        if unrecognized_keys:
            raise TypeError(f'Got an unexpected keyword argument(s): {unrecognized_keys}.')
        return _create_openai_tools_runnable(output_schema, llm, prompt=prompt, output_parser=output_parser, enforce_tool_usage=force_function_usage, first_tool_only=return_single)
    elif mode == 'openai-functions':
        return _create_openai_functions_structured_output_runnable(output_schema, llm, prompt=prompt, output_parser=output_parser, enforce_single_function_usage=force_function_usage, **kwargs)
    elif mode == 'openai-json':
        if force_function_usage:
            raise ValueError("enforce_single_function_usage is not supported for mode='openai-json'.")
        return _create_openai_json_runnable(output_schema, llm, prompt=prompt, output_parser=output_parser, **kwargs)
    else:
        raise ValueError(f"Invalid mode {mode}. Expected one of 'openai-tools', 'openai-functions', 'openai-json'.")