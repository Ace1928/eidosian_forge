import re
import ast
from ochat.evaluation.grading.math_grader import grade_answer
def fs_cothub_mmlu_match_answer(task_data, response):
    ans_line = response.split('answer is')
    if len(ans_line) == 1:
        return (False, '(C)')
    else:
        ans = ans_line[-1].strip()
    options = ['(A)', '(B)', '(C)', '(D)']
    for option in options:
        if option in ans:
            return (True, option)
    return (False, '(C)')