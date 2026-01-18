import re
import ast
from ochat.evaluation.grading.math_grader import grade_answer
def coding_humaneval_match_answer(task_data, response):

    def _function_exists(code, func_name):
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return True
        return False

    def _try_match(content, prefix, entrypoint):
        code_blocks = [m[1] for m in re.findall('(\\`{3}.*?\\n+)([\\s\\S]*?)(\\n+\\`{3})', content)] + [content]
        for block in code_blocks:
            try:
                code_completion = prefix + block
                if _function_exists(code_completion, entrypoint):
                    return code_completion
            except SyntaxError:
                pass
    humaneval_task = task_data['_metadata']
    include_prefix = humaneval_task['prompt'].split('def')[0].strip() + '\n\n'
    result = _try_match(response, include_prefix, humaneval_task['entry_point'])
    if result:
        return (True, {'task_id': humaneval_task['task_id'], 'completion': result})
    result = _try_match(response, humaneval_task['prompt'], humaneval_task['entry_point'])
    if result:
        return (True, {'task_id': humaneval_task['task_id'], 'completion': result})
    return (False, {'task_id': humaneval_task['task_id'], 'completion': response})