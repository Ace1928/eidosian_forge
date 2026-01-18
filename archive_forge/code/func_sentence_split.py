import json
import math
import re
import signal
from contextlib import contextmanager
from glob import glob
from os.path import join as pjoin
def sentence_split(st, max_len=120, max_sen=-1):
    pre_sentences = st.split('\n')
    res = []
    for pre_s in pre_sentences:
        add_sent = ' '.join([x + '</s>' if x in ['.', '!', '?'] or '.[' in x else x for x in pre_s.split()])
        pre_res = add_sent.split('</s>')
        pre_lst = []
        for sen in pre_res:
            if len(sen.split()) <= max_len:
                pre_lst += [sen]
            else:
                tab_sen = sen.split()
                while len(tab_sen) > max_len:
                    if ';' in tab_sen[:max_len]:
                        split_id = max_len - tab_sen[:max_len][::-1].index(';')
                        pre_lst += [' '.join(tab_sen[:split_id])]
                        tab_sen = tab_sen[split_id:]
                    elif '--' in tab_sen[:max_len]:
                        split_id = max_len - tab_sen[:max_len][::-1].index('--')
                        pre_lst += [' '.join(tab_sen[:split_id])]
                        tab_sen = tab_sen[split_id:]
                    else:
                        candidates = [w.count('.') == 1 for w in tab_sen[:max_len]]
                        if sum(candidates) > 0:
                            split_id = max_len - candidates[::-1].index(True) - 1
                            a, b = tab_sen[split_id].split('.')
                            pre_lst += [' '.join(tab_sen[:split_id] + [a])]
                            tab_sen = [b] + tab_sen[split_id + 1:]
                        else:
                            pre_lst += [' '.join(tab_sen[:max_len])]
                            tab_sen = tab_sen[max_len:]
                pre_lst += [' '.join(tab_sen)]
        res += pre_lst
    return [' '.join(s.split()) for s in res if len(s.strip()) > 0]