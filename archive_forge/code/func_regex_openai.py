from functools import singledispatch
from outlines.fsm.guide import RegexGuide
from outlines.generate.api import SequenceGenerator
from outlines.integrations.llamacpp import RegexLogitsProcessor
from outlines.models import OpenAI
from outlines.models.llamacpp import LlamaCpp, LlamaSequenceGenerator
from outlines.samplers import Sampler, multinomial
@regex.register(OpenAI)
def regex_openai(model: OpenAI, regex_str: str, sampler: Sampler=multinomial()):
    raise NotImplementedError('Cannot use regex-structured generation with an OpenAI model' + 'due to the limitations of the OpenAI API.')