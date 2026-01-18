from parlai.agents.transformer.transformer import TransformerClassifierAgent
from parlai.core.agents import create_agent, create_agent_from_shared
from parlai.tasks.dialogue_safety.agents import OK_CLASS, NOT_OK_CLASS
from parlai.utils.typing import TShared
import parlai.utils.logging as logging
import os
def find_all_offensive_language(self, text):
    """
        Find all offensive words from text in the filter.
        """
    if type(text) is str:
        toks = self.tokenize(text.lower())
    elif type(text) is list or type(text) is tuple:
        toks = text
    all_offenses = []
    for i in range(len(toks)):
        res = self._check_sequence(toks, i, self.offensive_trie)
        if res:
            all_offenses.append(res)
    return all_offenses