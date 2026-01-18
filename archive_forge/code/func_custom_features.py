import time
from typing import Any, Dict, List, Optional
from uuid import UUID
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks.utils import import_pandas
@property
def custom_features(self) -> list:
    """
        Define custom features for the model to automatically enrich the data with.
        Here, we enable the following enrichments:
        - Automatic Embedding generation for prompt and response
        - Text Statistics such as:
            - Automated Readability Index
            - Coleman Liau Index
            - Dale Chall Readability Score
            - Difficult Words
            - Flesch Reading Ease
            - Flesch Kincaid Grade
            - Gunning Fog
            - Linsear Write Formula
        - PII - Personal Identifiable Information
        - Sentiment Analysis

        """
    return [self.fdl.Enrichment(name='Prompt Embedding', enrichment='embedding', columns=[PROMPT]), self.fdl.TextEmbedding(name='Prompt CF', source_column=PROMPT, column='Prompt Embedding'), self.fdl.Enrichment(name='Response Embedding', enrichment='embedding', columns=[RESPONSE]), self.fdl.TextEmbedding(name='Response CF', source_column=RESPONSE, column='Response Embedding'), self.fdl.Enrichment(name='Text Statistics', enrichment='textstat', columns=[PROMPT, RESPONSE], config={'statistics': ['automated_readability_index', 'coleman_liau_index', 'dale_chall_readability_score', 'difficult_words', 'flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog', 'linsear_write_formula']}), self.fdl.Enrichment(name='PII', enrichment='pii', columns=[PROMPT, RESPONSE]), self.fdl.Enrichment(name='Sentiment', enrichment='sentiment', columns=[PROMPT, RESPONSE])]