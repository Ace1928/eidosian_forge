from typing import Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain.chains.base import Chain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.chains.constitutional_ai.principles import PRINCIPLES
from langchain.chains.constitutional_ai.prompts import CRITIQUE_PROMPT, REVISION_PROMPT
from langchain.chains.llm import LLMChain
@classmethod
def get_principles(cls, names: Optional[List[str]]=None) -> List[ConstitutionalPrinciple]:
    if names is None:
        return list(PRINCIPLES.values())
    else:
        return [PRINCIPLES[name] for name in names]