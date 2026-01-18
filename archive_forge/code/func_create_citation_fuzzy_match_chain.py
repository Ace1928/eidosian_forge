from typing import Iterator, List
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers.openai_functions import PydanticOutputFunctionsParser
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions.utils import get_llm_kwargs
def create_citation_fuzzy_match_chain(llm: BaseLanguageModel) -> LLMChain:
    """Create a citation fuzzy match chain.

    Args:
        llm: Language model to use for the chain.

    Returns:
        Chain (LLMChain) that can be used to answer questions with citations.
    """
    output_parser = PydanticOutputFunctionsParser(pydantic_schema=QuestionAnswer)
    schema = QuestionAnswer.schema()
    function = {'name': schema['title'], 'description': schema['description'], 'parameters': schema}
    llm_kwargs = get_llm_kwargs(function)
    messages = [SystemMessage(content='You are a world class algorithm to answer questions with correct and exact citations.'), HumanMessage(content='Answer question using the following context'), HumanMessagePromptTemplate.from_template('{context}'), HumanMessagePromptTemplate.from_template('Question: {question}'), HumanMessage(content='Tips: Make sure to cite your sources, and use the exact words from the context.')]
    prompt = ChatPromptTemplate(messages=messages)
    chain = LLMChain(llm=llm, prompt=prompt, llm_kwargs=llm_kwargs, output_parser=output_parser)
    return chain