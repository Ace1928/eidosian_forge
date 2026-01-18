import random
import io
import os
import pickle
from parlai.utils.misc import msg_to_str
def add_negs(msg, d, ind, label_type, split, num_cands, use_affordances):
    if label_type == 'emote':
        msg['label_candidates'] = cands['emote']
    if label_type == 'which':
        msg['label_candidates'] = cands['which']
    if label_type == 'action':
        if use_affordances:
            msg['label_candidates'] = d['available_actions'][ind]
        else:
            msg['label_candidates'] = d['no_affordance_actions'][ind]
    if label_type == 'speech':
        cnt = 0
        label = msg['labels']
        negs = []
        used = {}
        while True:
            ind = rand.randint(0, len(cands['speech']) - 1)
            txt = cands['speech'][ind]
            if txt != label and ind not in used:
                negs.append(txt)
                used[ind] = True
                cnt += 1
                if cnt == num_cands - 1:
                    break
        negs.insert(rand.randrange(len(negs) + 1), label)
        msg['label_candidates'] = negs