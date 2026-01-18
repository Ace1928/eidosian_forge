import random
import io
import os
import pickle
from parlai.utils.misc import msg_to_str
def gen_no_affordance_actions_for_dialogue(d):
    characters = [d['agents'][0]['name'], d['agents'][1]['name']]
    other_char = 1
    no_affordance_actions = []
    last_carry = []
    for idx in range(len(d['speech'])):
        char = characters[other_char]
        curr_carry = d['carrying'][idx] + d['wearing'][idx] + d['wielding'][idx]
        no_affordance_actions_turn = get_no_affordance_actions(d['room_objects'][idx], curr_carry, last_carry, char)
        no_affordance_actions_turn += d['available_actions'][idx]
        no_affordance_actions.append(list(set(no_affordance_actions_turn)))
        last_carry = curr_carry
        other_char = 1 - other_char
    d['no_affordance_actions'] = no_affordance_actions