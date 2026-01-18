import random
import io
import os
import pickle
from parlai.utils.misc import msg_to_str
def get_no_affordance_actions(in_room, carrying, other_carrying, other_name):
    all_actions = []
    in_room = [i[2:] for i in in_room]
    carrying = [i[2:] for i in carrying]
    other_carrying = [i[2:] for i in other_carrying]
    for obj in other_carrying:
        all_actions.append('steal {} from {}'.format(obj, other_name))
    for obj in in_room:
        all_actions.append('get {}'.format(obj))
    for obj in carrying:
        for f in ['drop {}', 'wield {}', 'wear {}', 'remove {}', 'eat {}', 'drink {}']:
            all_actions.append(f.format(obj))
        all_actions.append('give {} to {}'.format(obj, other_name))
        for obj2 in in_room:
            all_actions.append('put {} in {}'.format(obj, obj2))
        for obj2 in carrying:
            if obj2 == obj:
                continue
            all_actions.append('put {} in {}'.format(obj, obj2))
    return list(set(all_actions))