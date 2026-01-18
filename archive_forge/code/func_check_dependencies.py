from typing import Dict, List
import numpy as np
from langchain_core.example_selectors import BaseExampleSelector
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, root_validator
@root_validator(pre=True)
def check_dependencies(cls, values: Dict) -> Dict:
    """Check that valid dependencies exist."""
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except ImportError as e:
        raise ImportError('Not all the correct dependencies for this ExampleSelect exist.Please install nltk with `pip install nltk`.') from e
    return values