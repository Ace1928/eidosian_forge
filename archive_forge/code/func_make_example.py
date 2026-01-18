import json
import os
from os.path import join as pjoin
from os.path import isfile, isdir
from parlai.core.params import ParlaiParser
from time import time
from data_utils import sentence_split, tf_idf_vec, tf_idf_dist
def make_example(qa_dct, docs_list, word_freqs, n_sents=100, n_context=3):
    q_id = qa_dct['id']
    question = qa_dct['title'][0] + ' --T-- ' + qa_dct['selftext'][0]
    answer = qa_dct['comments'][0]['body'][0]
    document = select_pars(qa_dct, docs_list, word_freqs, n_sents, n_context)
    return {'id': q_id, 'question': question, 'document': document, 'answer': answer}