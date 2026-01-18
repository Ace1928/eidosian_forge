import os, copy, types, gc, sys
import numpy as np
from prompt_toolkit import prompt
import torch
from rwkv.model import RWKV
from rwkv.utils import PIPELINE
def load_prompt(PROMPT_FILE):
    variables = {}
    with open(PROMPT_FILE, 'rb') as file:
        exec(compile(file.read(), PROMPT_FILE, 'exec'), variables)
    user, bot, interface, init_prompt = (variables['user'], variables['bot'], variables['interface'], variables['init_prompt'])
    init_prompt = init_prompt.strip().split('\n')
    for c in range(len(init_prompt)):
        init_prompt[c] = init_prompt[c].strip().strip('\u3000').strip('\r')
    init_prompt = '\n' + '\n'.join(init_prompt).strip() + '\n\n'
    return (user, bot, interface, init_prompt)