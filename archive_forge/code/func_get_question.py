import unittest
import uuid
import datetime
from boto.mturk.question import (
from ._init_environment import SetHostMTurkConnection, config_environment
@staticmethod
def get_question():
    qn_content = QuestionContent()
    qn_content.append_field('Title', 'Boto no hit type question content')
    qn_content.append_field('Text', 'What is a boto no hit type?')
    qn = Question(identifier=str(uuid.uuid4()), content=qn_content, answer_spec=AnswerSpecification(FreeTextAnswer()))
    return qn