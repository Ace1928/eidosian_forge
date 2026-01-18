from the model zoo.
from parlai.core.agents import Agent, create_agent, create_agent_from_shared
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE
from parlai.zoo.wizard_of_wikipedia.knowledge_retriever import download
import json
import os
def get_passages(self, act):
    """
        Format passages retrieved by taking the first paragraph of the top
        `num_retrieved` passages.
        """
    retrieved_txt = act.get('text', '')
    cands = act.get('text_candidates', [])
    if len(cands) > 0:
        retrieved_txts = cands[:self.opt['num_retrieved']]
    else:
        retrieved_txts = [retrieved_txt]
    retrieved_txt_format = []
    retrieved_txt_format_no_title = []
    for ret_txt in retrieved_txts:
        paragraphs = ret_txt.split('\n')
        if len(paragraphs) > 2:
            sentences = self.sent_tok.tokenize(paragraphs[2])
            for sent in sentences:
                if self.add_token_knowledge:
                    delim = ' ' + TOKEN_KNOWLEDGE + ' '
                else:
                    delim = ' '
                retrieved_txt_format.append(delim.join([paragraphs[0], sent]))
                retrieved_txt_format_no_title.append(sent)
    if len(retrieved_txt_format) > 0:
        passages = '\n'.join(retrieved_txt_format)
        passages_no_title = '\n'.join(retrieved_txt_format_no_title)
    else:
        passages = ''
        passages_no_title = ''
    return (passages, passages_no_title)