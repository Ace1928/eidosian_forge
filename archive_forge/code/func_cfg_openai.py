from functools import singledispatch
from outlines.fsm.guide import CFGGuide
from outlines.generate.api import SequenceGenerator
from outlines.models import OpenAI
from outlines.models.llamacpp import (
from outlines.samplers import Sampler, multinomial
@cfg.register(OpenAI)
def cfg_openai(model, cfg_str: str, sampler: Sampler=multinomial()):
    raise NotImplementedError('Cannot use grammar-structured generation with an OpenAI model' + 'due to the limitations of the OpenAI API.')