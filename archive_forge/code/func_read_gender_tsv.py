from parlai.core.message import Message
from parlai.core.teachers import FixedDialogTeacher
from parlai.tasks.light_dialog.agents import DefaultTeacher as OrigLightTeacher
from parlai.tasks.light_genderation_bias.build import build
from collections import deque
from copy import deepcopy
import csv
import json
import os
import random
def read_gender_tsv(path, remove_verbs=True):
    """
    Load TSV of gendered word lists and return a dict.
    """
    gender_dct = {}
    with open(path) as tsvfile:
        reader = list(csv.reader(tsvfile, delimiter='\t'))
        title_lst = reader[0]
        title_dict = {}
        for idx, title in enumerate(title_lst):
            title_dict[idx] = title
        for i in range(1, len(reader)):
            row = reader[i]
            word = row[0].lower()
            gender_dct[word] = {}
            for j, category in enumerate(row[1:]):
                gender_dct[word][title_dict[j + 1]] = category
    if remove_verbs:
        return {k: v for k, v in gender_dct.items() if v['syncategory'] != 'verb'}
    return gender_dct