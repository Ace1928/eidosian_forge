from .agents import BASE_PYTHON_TOOLS, clean_code_for_chat
from .python_interpreter import InterpretorError, evaluate
def get_theoretical_tools(agent_answer, theoretical_answer, code_answer):
    if not isinstance(theoretical_answer, list):
        return {name for name in TEST_TOOLS if name in code_answer}
    if isinstance(agent_answer, dict):
        for one_answer, one_code in zip(theoretical_answer, code_answer):
            if one_answer in agent_answer.values():
                return {name for name in TEST_TOOLS if name in one_code}
    for one_answer, one_code in zip(theoretical_answer, code_answer):
        if agent_answer == one_answer:
            return {name for name in TEST_TOOLS if name in one_code}
    return {name for name in TEST_TOOLS if name in code_answer[0]}