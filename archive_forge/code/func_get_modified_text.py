import copy
import json
import os
import random
import re
from collections import defaultdict
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm
from parlai.core.opt import Opt
from parlai.core.teachers import (
from parlai.tasks.convai2.agents import (
from parlai.tasks.empathetic_dialogues.agents import EmpatheticDialoguesTeacher
from parlai.tasks.wizard_of_wikipedia.agents import WizardDialogKnowledgeTeacher
from parlai.utils.misc import warn_once
from .build import build
def get_modified_text(self, text):
    has_neither = not self.should_have_personas and (not self.should_have_topics)
    has_wow_topic_only = not self.should_have_personas and self.should_have_topics
    has_persona_only = not self.should_have_topics and self.should_have_personas
    if self.should_have_personas and (has_neither or has_wow_topic_only) or (self.should_have_topics and (has_neither or has_persona_only)):
        raise Exception(f'Malformed text: {text}, should_have_personas: {self.should_have_personas}, should_have_topics: {self.should_have_topics}, has_neither: {has_neither}, has_wow_topic_only: {has_wow_topic_only}, has_persona_only: {has_persona_only}')
    if has_neither:
        persona = self.__choose_persona_from_text(text)
        topic = self.__choose_topic(persona)
        utt = text
    elif has_wow_topic_only:
        parts = text.strip().split('\n')
        if len(parts) > 1:
            topic = parts[0] + '\n'
            utt = parts[1]
            persona = self.__choose_persona_from_topic(topic)
        else:
            topic = parts[0] + '\n'
            utt = ''
            persona = self.__choose_persona_from_topic(topic)
    elif has_persona_only:
        lines = text.strip().split('\n')
        utt = lines[-1]
        persona = ''.join((l + '\n' for l in lines[:-1]))
        topic = self.__choose_topic(persona)
    else:
        raise Exception(f'Unknown structure of utterance: {text}')
    modified_utterance = persona + topic + utt
    return modified_utterance