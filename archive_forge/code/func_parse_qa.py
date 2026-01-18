from parlai.core.teachers import DialogTeacher
from .build import build
import os
import copy
def parse_qa(qa_line):
    qa_split = qa_line.split('\t?\t')
    question = context + '\n' + qa_split[0].replace('\t_', '').replace('\t', ' ') + '?'
    answers = qa_split[1].split(' ### ')
    return [question, answers]