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
def display_messages(msgs: List[Dict[str, Any]], prettify: bool=False, ignore_fields: str='', max_len: int=1000, verbose: bool=False) -> Optional[str]:
    """
    Return a string describing the set of messages provided.

    If prettify is true, candidates are displayed using prettytable. ignore_fields
    provides a list of fields in the msgs which should not be displayed.
    """

    def _token_losses_line(msg: Dict[str, Any], ignore_fields: List[str], space: str) -> Optional[str]:
        """
        Displays the loss associated with each token. Can be used for debugging
        generative models.

        See TorchGeneratorAgent._construct_token_losses for an example implementation.
        """
        key = 'token_losses'
        token_losses = msg.get(key, None)
        if key in ignore_fields or not token_losses:
            return None
        formatted_tl = ' | '.join([f'{tl[0]} {float('{:.4g}'.format(tl[1]))}' for tl in token_losses])
        return f'{space}[{key}]: {formatted_tl}'
    lines = []
    episode_done = False
    ignore_fields_ = ignore_fields.split(',')
    for index, msg in enumerate(msgs):
        if msg is None or (index == 1 and 'agent_reply' in ignore_fields_):
            continue
        agent_id = msg.get('id', '[no id field]')
        if verbose:
            lines.append(colorize('[id]:', 'field') + ' ' + colorize(agent_id, 'id'))
        if msg.get('episode_done'):
            episode_done = True
        space = ''
        if len(msgs) == 2 and index == 1:
            space = '   '
        if msg.get('reward', 0) != 0:
            lines.append(space + '[reward: {r}]'.format(r=msg['reward']))
        for key in msg:
            if key not in DISPLAY_MESSAGE_DEFAULT_FIELDS and key not in ignore_fields_:
                field = colorize('[' + key + ']:', 'field')
                if type(msg[key]) is list:
                    value = _ellipse(msg[key], sep='\n  ')
                else:
                    value = clip_text(str(msg.get(key)), max_len)
                line = field + ' ' + colorize(value, 'text2')
                lines.append(space + line)
        if type(msg.get('image')) in [str, torch.Tensor]:
            lines.append(f'[ image ]: {msg['image']}')
        if msg.get('text', ''):
            text = clip_text(msg['text'], max_len)
            if index == 0:
                style = 'bold_text'
            else:
                style = 'labels'
            if verbose:
                lines.append(space + colorize('[text]:', 'field') + ' ' + colorize(text, style))
            else:
                lines.append(space + colorize('[' + agent_id + ']:', 'field') + ' ' + colorize(text, style))
        for field in {'labels', 'eval_labels', 'label_candidates', 'text_candidates'}:
            if msg.get(field) and field not in ignore_fields_:
                string = '{}{} {}'.format(space, colorize('[' + field + ']:', 'field'), colorize(_ellipse(msg[field]), field))
                lines.append(string)
        token_loss_line = _token_losses_line(msg, ignore_fields_, space)
        if token_loss_line:
            lines.append(token_loss_line)
    if episode_done:
        lines.append(colorize('- - - - - - - END OF EPISODE - - - - - - - - - -', 'highlight'))
    return '\n'.join(lines)