from __future__ import annotations
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast
from langchain_core.callbacks import (
from langchain_core.language_models import (
from langchain_core.load.dump import dumpd
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import BaseLLMOutputParser, StrOutputParser
from langchain_core.outputs import ChatGeneration, Generation, LLMResult
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import Extra, Field
from langchain_core.runnables import (
from langchain_core.runnables.configurable import DynamicRunnable
from langchain_core.utils.input import get_colored_text
from langchain.chains.base import Chain
def prep_prompts(self, input_list: List[Dict[str, Any]], run_manager: Optional[CallbackManagerForChainRun]=None) -> Tuple[List[PromptValue], Optional[List[str]]]:
    """Prepare prompts from inputs."""
    stop = None
    if len(input_list) == 0:
        return ([], stop)
    if 'stop' in input_list[0]:
        stop = input_list[0]['stop']
    prompts = []
    for inputs in input_list:
        selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
        prompt = self.prompt.format_prompt(**selected_inputs)
        _colored_text = get_colored_text(prompt.to_string(), 'green')
        _text = 'Prompt after formatting:\n' + _colored_text
        if run_manager:
            run_manager.on_text(_text, end='\n', verbose=self.verbose)
        if 'stop' in inputs and inputs['stop'] != stop:
            raise ValueError('If `stop` is present in any inputs, should be present in all.')
        prompts.append(prompt)
    return (prompts, stop)