from parlai.core.worlds import validate
from parlai.mturk.core.worlds import MTurkOnboardWorld, MTurkTaskWorld
import parlai.mturk.core.mturk_utils as mturk_utils
import random
def generate_questions(self, num):
    questions = []
    for _ in range(num):
        num1 = random.randint(1, 20)
        num2 = random.randint(3, 16)
        questions.append(['What is {} + {}?'.format(num1, num2), '{}'.format(num1 + num2)])
    return questions