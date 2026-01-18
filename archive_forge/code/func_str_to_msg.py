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
def str_to_msg(txt, ignore_fields=''):
    """
    Convert formatted string to ParlAI message dict.

    :param txt:
        formatted string to convert. String format is tab-separated fields,
        with colon separating field name and contents.
    :param ignore_fields:
        (default '') comma-separated field names to not
        include in the msg dict even if they're in the string.
    """

    def tostr(txt):
        txt = str(txt)
        txt = txt.replace('\\t', '\t')
        txt = txt.replace('\\n', '\n')
        txt = txt.replace('__PIPE__', '|')
        return txt

    def tolist(txt):
        vals = txt.split('|')
        for v in vals:
            v = tostr(v)
        return vals

    def convert(key, value):
        if key == 'text' or key == 'id':
            return tostr(value)
        elif key == 'label_candidates' or key == 'labels' or key == 'eval_labels' or (key == 'text_candidates'):
            return tolist(value)
        elif key == 'episode_done':
            return bool(value)
        else:
            return tostr(value)
    if txt == '' or txt is None:
        return None
    msg = {}
    for t in txt.split('\t'):
        ind = t.find(':')
        key = t[:ind]
        value = t[ind + 1:]
        if key not in ignore_fields.split(','):
            msg[key] = convert(key, value)
    msg['episode_done'] = msg.get('episode_done', False)
    return Message(msg)