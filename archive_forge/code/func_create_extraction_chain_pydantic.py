from typing import Any, List, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers.openai_functions import (
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions.utils import (
def create_extraction_chain_pydantic(pydantic_schema: Any, llm: BaseLanguageModel, prompt: Optional[BasePromptTemplate]=None, verbose: bool=False) -> Chain:
    """Creates a chain that extracts information from a passage using pydantic schema.

    Args:
        pydantic_schema: The pydantic schema of the entities to extract.
        llm: The language model to use.
        prompt: The prompt to use for extraction.
        verbose: Whether to run in verbose mode. In verbose mode, some intermediate
            logs will be printed to the console. Defaults to the global `verbose` value,
            accessible via `langchain.globals.get_verbose()`

    Returns:
        Chain that can be used to extract information from a passage.
    """

    class PydanticSchema(BaseModel):
        info: List[pydantic_schema]
    openai_schema = pydantic_schema.schema()
    openai_schema = _resolve_schema_references(openai_schema, openai_schema.get('definitions', {}))
    function = _get_extraction_function(openai_schema)
    extraction_prompt = prompt or ChatPromptTemplate.from_template(_EXTRACTION_TEMPLATE)
    output_parser = PydanticAttrOutputFunctionsParser(pydantic_schema=PydanticSchema, attr_name='info')
    llm_kwargs = get_llm_kwargs(function)
    chain = LLMChain(llm=llm, prompt=extraction_prompt, llm_kwargs=llm_kwargs, output_parser=output_parser, verbose=verbose)
    return chain