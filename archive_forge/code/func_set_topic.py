import json
import os
from random import choice
from typing import Dict
from parlai.core.agents import create_agent, add_datapath_and_model_args
from parlai.core.message import Message
from parlai.core.params import ParlaiParser
from parlai.tasks.wizard_of_wikipedia.build import build
from projects.wizard_of_wikipedia.knowledge_retriever.knowledge_retriever import KnowledgeRetrieverAgent
from openchat.base import ParlaiGenerationAgent
def set_topic(self, topic):
    if topic.lower() == 'random':
        self.chosen_topic = choice(self.topic_list)
    else:
        assert topic in self.topic_list, f'Wrong topic: {topic}, You can check available topic using `available_topics()`'
        self.chosen_topic = topic