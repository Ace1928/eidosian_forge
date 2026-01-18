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
def create_structured_output_chain(output_schema: Union[Dict[str, Any], Type[BaseModel]], llm: BaseLanguageModel, prompt: BasePromptTemplate, *, output_key: str='function', output_parser: Optional[BaseLLMOutputParser]=None, **kwargs: Any) -> LLMChain:
    '''[Legacy] Create an LLMChain that uses an Ernie function to get a structured output.

    Args:
        output_schema: Either a dictionary or pydantic.BaseModel class. If a dictionary
            is passed in, it's assumed to already be a valid JsonSchema.
            For best results, pydantic.BaseModels should have docstrings describing what
            the schema represents and descriptions for the parameters.
        llm: Language model to use, assumed to support the Ernie function-calling API.
        prompt: BasePromptTemplate to pass to the model.
        output_key: The key to use when returning the output in LLMChain.__call__.
        output_parser: BaseLLMOutputParser to use for parsing model outputs. By default
            will be inferred from the function types. If pydantic.BaseModels are passed
            in, then the OutputParser will try to parse outputs using those. Otherwise
            model outputs will simply be parsed as JSON.

    Returns:
        An LLMChain that will pass the given function to the model.

    Example:
        .. code-block:: python

                from typing import Optional

                from langchain.chains.ernie_functions import create_structured_output_chain
                from langchain_community.chat_models import ErnieBotChat
                from langchain.prompts import ChatPromptTemplate

                from langchain.pydantic_v1 import BaseModel, Field

                class Dog(BaseModel):
                    """Identifying information about a dog."""

                    name: str = Field(..., description="The dog's name")
                    color: str = Field(..., description="The dog's color")
                    fav_food: Optional[str] = Field(None, description="The dog's favorite food")

                llm = ErnieBotChat(model_name="ERNIE-Bot-4")
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("user", "Use the given format to extract information from the following input: {input}"),
                        ("assistant", "OK!"),
                        ("user", "Tip: Make sure to answer in the correct format"),
                    ]
                )
                chain = create_structured_output_chain(Dog, llm, prompt)
                chain.run("Harry was a chubby brown beagle who loved chicken")
                # -> Dog(name="Harry", color="brown", fav_food="chicken")
    '''
    if isinstance(output_schema, dict):
        function: Any = {'name': 'output_formatter', 'description': 'Output formatter. Should always be used to format your response to the user.', 'parameters': output_schema}
    else:

        class _OutputFormatter(BaseModel):
            """Output formatter. Should always be used to format your response to the user."""
            output: output_schema
        function = _OutputFormatter
        output_parser = output_parser or PydanticAttrOutputFunctionsParser(pydantic_schema=_OutputFormatter, attr_name='output')
    return create_ernie_fn_chain([function], llm, prompt, output_key=output_key, output_parser=output_parser, **kwargs)