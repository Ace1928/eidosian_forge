from parlai.core.teachers import DialogTeacher
from .build import build
import os
def extract_qa(qa_data):
    line_data = qa_data.split('\t')
    question_type, anon_question, deanon, context = line_data[:4]
    answer = line_data[4:]
    if answer == []:
        answer = ['No answer']
    return (context + '\n' + anon_question.replace('XXX', deanon), answer)