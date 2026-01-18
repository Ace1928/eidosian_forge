import copy
from parlai.core.teachers import FixedDialogTeacher, MultiTaskTeacher
from .build import build
import json
import os
import random
def extract_passage_and_text(self, data, idx):
    passage_key = _first_key(data['dialog'][idx]['checked_sentence'])
    dialog_entry = data['dialog'][idx]
    text = passage = None
    if 'chosen' in passage_key:
        passage = ' '.join(data['chosen_topic_passage'])
        text = data['chosen_topic']
    elif 'self' in passage_key:
        passages = data['dialog'][idx - 2]['retrieved_passages']
        passage = None
        key = _first_val(dialog_entry['checked_passage'])
        for p in passages:
            if key in p:
                passage = ' '.join(p[key])
                break
        text = data['dialog'][idx - 2]['text']
    elif 'partner' in passage_key:
        passages = data['dialog'][idx - 1]['retrieved_passages']
        passage = None
        key = _first_val(dialog_entry['checked_passage'])
        for p in passages:
            if key in p:
                passage = ' '.join(p[key])
                break
        text = data['dialog'][idx - 1]['text']
    return (passage, text)