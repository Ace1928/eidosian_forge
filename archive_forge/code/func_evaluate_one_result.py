from .agents import BASE_PYTHON_TOOLS, clean_code_for_chat
from .python_interpreter import InterpretorError, evaluate
def evaluate_one_result(explanation, code, agent_answer, theoretical_answer, answer, verbose=False):
    tools_in_explanation = {name for name in TEST_TOOLS if f'`{name}`' in explanation}
    theoretical_tools = get_theoretical_tools(agent_answer, theoretical_answer, answer)
    if tools_in_explanation == theoretical_tools:
        tool_selection_score = 1.0
        tool_selection_errors = None
    else:
        missing_tools = len(theoretical_tools - tools_in_explanation)
        unexpected_tools = len(tools_in_explanation - theoretical_tools)
        tool_selection_score = max(0, 1.0 - 0.25 * missing_tools - 0.25 * unexpected_tools)
        tool_selection_errors = {'selected_tools': tools_in_explanation, 'theoretical_tools': theoretical_tools}
    tools_in_code = {name for name in TEST_TOOLS if name in code}
    if tools_in_code == theoretical_tools:
        tool_used_score = 1.0
        tool_used_errors = None
    else:
        missing_tools = len(theoretical_tools - tools_in_code)
        unexpected_tools = len(tools_in_code - theoretical_tools)
        tool_used_score = max(0, 1.0 - 0.25 * missing_tools - 0.25 * unexpected_tools)
        tool_used_errors = {'selected_tools': tools_in_explanation, 'theoretical_tools': theoretical_tools}
    score = score_code(agent_answer, theoretical_answer, verbose=verbose)
    if score < 1.0:
        code_errors = {'code_produced': code, 'evaluation': agent_answer, 'theoretical_answer': theoretical_answer}
    else:
        code_errors = None
    return ((tool_selection_score, tool_used_score, score), (tool_selection_errors, tool_used_errors, code_errors))