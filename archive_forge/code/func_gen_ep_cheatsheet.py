from .build import build
import os
def gen_ep_cheatsheet(convo):
    """
    Generates a cheatsheet for a particular conversation (episode).
    The index and it's significance in the cheatsheet is shown below:
        0: First index of a USER that has an ASSISTANT reply to it
        1: Last index of a USER that has an ASSISTANT reply to it
        2: First index of an ASSISTANT that has a USER reply to it
        3: Last index of an ASSISTANT that has a USER reply to it
        4: Number of examples for USER speech  as text and ASSISTANT speech as label
        5: Number of examples for ASSISTANT speech as text and USER speech  as label
    :param convo:
        The dialogue between USER and ASSISTANT [after smoothening]
    """
    cheatsheet = [-1, -1, -1, -1, 0, 0]
    for idx in range(1, len(convo)):
        if convo[idx - 1]['speaker'] == 'USER' and convo[idx]['speaker'] == 'ASSISTANT':
            if cheatsheet[0] == -1:
                cheatsheet[0] = idx - 1
            cheatsheet[1] = idx - 1
        if convo[idx - 1]['speaker'] == 'ASSISTANT' and convo[idx]['speaker'] == 'USER':
            if cheatsheet[2] == -1:
                cheatsheet[2] = idx - 1
            cheatsheet[3] = idx - 1
        if cheatsheet[1] != -1:
            cheatsheet[4] = (cheatsheet[1] - cheatsheet[0]) // 2 + 1
        if cheatsheet[3] != -1:
            cheatsheet[5] = (cheatsheet[3] - cheatsheet[2]) // 2 + 1
    return cheatsheet