import copy
from parlai.core.teachers import FixedDialogTeacher, MultiTaskTeacher
from .build import build
import json
import os
import random
def get_span_label(self, data, idx):
    dialog_entry = data['dialog'][idx]
    said = dialog_entry['text']
    sentence = _first_val(dialog_entry['checked_sentence'])
    overlap = self.get_span(said, sentence)
    if not overlap or overlap in self.stop_words:
        label = sentence
    else:
        label = overlap
    return label