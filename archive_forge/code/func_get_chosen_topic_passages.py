from the model zoo.
from parlai.core.agents import Agent, create_agent, create_agent_from_shared
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE
from parlai.zoo.wizard_of_wikipedia.knowledge_retriever import download
import json
import os
def get_chosen_topic_passages(self, chosen_topic):
    retrieved_txt_format = []
    retrieved_txt_format_no_title = []
    if chosen_topic in self.wiki_map:
        retrieved_txt = self.wiki_map[chosen_topic]
        retrieved_txts = retrieved_txt.split('\n')
        if len(retrieved_txts) > 1:
            combined = ' '.join(retrieved_txts[2:])
            sentences = self.sent_tok.tokenize(combined)
            total = 0
            for sent in sentences:
                if total >= 10:
                    break
                if len(sent) > 0:
                    if self.add_token_knowledge:
                        delim = ' ' + TOKEN_KNOWLEDGE + ' '
                    else:
                        delim = ' '
                    retrieved_txt_format.append(delim.join([chosen_topic, sent]))
                    retrieved_txt_format_no_title.append(sent)
                    total += 1
    if len(retrieved_txt_format) > 0:
        passages = '\n'.join(retrieved_txt_format)
        passages_no_title = '\n'.join(retrieved_txt_format_no_title)
    else:
        passages = ''
        passages_no_title = ''
    return (passages, passages_no_title)