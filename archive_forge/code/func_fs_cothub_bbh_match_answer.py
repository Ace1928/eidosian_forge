import re
import ast
from ochat.evaluation.grading.math_grader import grade_answer
def fs_cothub_bbh_match_answer(task_data, response):
    ans_line = response.split('answer is ')
    if len(ans_line) == 1:
        return (False, response)
    else:
        ans = ans_line[-1].strip()
    if task_data['options']:
        options = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)', '(L)', '(M)', '(N)', '(O)', '(P)', '(Q)', '(R)', '(S)', '(T)', '(U)', '(V)', '(W)', '(X)', '(Y)', '(Z)']
        for option in options:
            if option in ans:
                return (True, option)
        return (False, ans)
    else:
        if len(ans) and ans[-1] == '.':
            ans = ans[:-1]
        return (True, ans)