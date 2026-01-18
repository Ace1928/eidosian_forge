from typing import Dict, List
import numpy as np
from langchain_core.example_selectors import BaseExampleSelector
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, root_validator
def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
    """Return list of examples sorted by ngram_overlap_score with input.

        Descending order.
        Excludes any examples with ngram_overlap_score less than or equal to threshold.
        """
    inputs = list(input_variables.values())
    examples = []
    k = len(self.examples)
    score = [0.0] * k
    first_prompt_template_key = self.example_prompt.input_variables[0]
    for i in range(k):
        score[i] = ngram_overlap_score(inputs, [self.examples[i][first_prompt_template_key]])
    while True:
        arg_max = np.argmax(score)
        if score[arg_max] < self.threshold or abs(score[arg_max] - self.threshold) < 1e-09:
            break
        examples.append(self.examples[arg_max])
        score[arg_max] = self.threshold - 1.0
    return examples