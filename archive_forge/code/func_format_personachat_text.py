from parlai.core.agents import create_agent_from_shared
from parlai.mturk.core.legacy_2018.agents import TIMEOUT_MESSAGE
from parlai.core.worlds import validate, MultiAgentDialogWorld
from parlai.mturk.core.legacy_2018.worlds import MTurkOnboardWorld
from parlai.core.message import Message
from parlai.utils.strings import normalize_reply
from joblib import Parallel, delayed
import numpy as np
import os
import json
import random
import time
import torch
import copy
def format_personachat_text(self, text):
    new_text = text.lower()
    switch_list = [("we're", 'were'), ("let's", 'lets'), ("it's", 'its'), ("who's", 'whos'), ("you're", 'youre'), ("you've", 'youve'), ("he'd", 'hed'), ("he'll", 'hell')]
    for tup in switch_list:
        new_text = new_text.replace(tup[0], tup[1])
    return new_text