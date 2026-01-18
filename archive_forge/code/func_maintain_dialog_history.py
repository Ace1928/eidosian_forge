from collections import deque, OrderedDict
from typing import Union, Optional, Set, Any, Dict, List, Tuple
from datetime import timedelta
import functools
import math
import time
import re
import shutil
import json
from parlai.core.message import Message
from parlai.utils.strings import colorize
import parlai.utils.logging as logging
def maintain_dialog_history(history, observation, reply='', historyLength=1, useReplies='label_else_model', dict=None, useStartEndIndices=True, splitSentences=False):
    """
    Keep track of dialog history, up to a truncation length.

    Either includes replies from the labels, model, or not all using param
    'replies'.

    DEPRECATED. USE PARLAI.CORE.TORCH_AGENT INSTEAD.
    """

    def parse(txt, splitSentences):
        if dict is not None:
            if splitSentences:
                vec = [dict.txt2vec(t) for t in txt.split('\n')]
            else:
                vec = dict.txt2vec(txt)
            return vec
        else:
            return [txt]
    if 'dialog' not in history:
        history['dialog'] = deque(maxlen=historyLength)
        history['episode_done'] = False
        history['labels'] = []
    if history['episode_done']:
        history['dialog'].clear()
        history['labels'] = []
        useReplies = 'none'
        history['episode_done'] = False
    if useReplies != 'none':
        if useReplies == 'model' or (useReplies == 'label_else_model' and len(history['labels']) == 0):
            if reply:
                if useStartEndIndices:
                    reply = dict.start_token + ' ' + reply
                history['dialog'].extend(parse(reply, splitSentences))
        elif len(history['labels']) > 0:
            r = history['labels'][0]
            history['dialog'].extend(parse(r, splitSentences))
    obs = observation
    if 'text' in obs:
        if useStartEndIndices:
            obs['text'] = dict.end_token + ' ' + obs['text']
        history['dialog'].extend(parse(obs['text'], splitSentences))
    history['episode_done'] = obs['episode_done']
    labels = obs.get('labels', obs.get('eval_labels', None))
    if labels is not None:
        if useStartEndIndices:
            history['labels'] = [dict.start_token + ' ' + l for l in labels]
        else:
            history['labels'] = labels
    return history['dialog']