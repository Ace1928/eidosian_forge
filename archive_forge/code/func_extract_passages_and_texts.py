import copy
from parlai.core.teachers import FixedDialogTeacher, MultiTaskTeacher
from .build import build
import json
import os
import random
def extract_passages_and_texts(self, d, idx):
    chosen_passages = ' '.join(d['chosen_topic_passage'])
    chosen_text = d['chosen_topic']
    if idx - 1 >= 0:
        appr_passages = d['dialog'][idx - 1]['retrieved_passages']
        appr_text = d['dialog'][idx - 1]['text']
        appr_list = []
        for passage in appr_passages:
            for v in passage.values():
                temp = ' '.join(v)
                appr_list.append(temp)
        appr = '\n'.join(appr_list)
    else:
        appr_passages = ''
        appr_text = ''
    if idx - 2 >= 0:
        wizard_passages = d['dialog'][idx - 2]['retrieved_passages']
        wizard_text = d['dialog'][idx - 2]['text']
        wizard_list = []
        for passage in wizard_passages:
            for v in passage.values():
                temp = ' '.join(v)
                wizard_list.append(temp)
        wizard = '\n'.join(wizard_list)
    else:
        wizard_passages = ''
        wizard_text = ''
    if idx - 2 >= 0:
        passages = '\n'.join([chosen_passages, wizard, appr])
        texts = ' '.join([chosen_text, wizard_text, appr_text])
    elif idx - 1 >= 0:
        passages = '\n'.join([chosen_passages, appr])
        texts = ' '.join([chosen_text, appr_text])
    else:
        passages = chosen_passages
        texts = chosen_text
    return (passages, texts)