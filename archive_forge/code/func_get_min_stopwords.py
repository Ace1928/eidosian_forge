import copy
from parlai.core.teachers import FixedDialogTeacher, MultiTaskTeacher
from .build import build
import json
import os
import random
def get_min_stopwords(self, word_set):
    min_count = 1000000000000
    min_words = ''
    for words in word_set:
        count = 0
        for stop in self.stop_words:
            if stop in words:
                count += 1
        if count < min_count:
            min_count = count
            min_words = words
    return min_words