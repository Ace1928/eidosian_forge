from parso.python import tree
from parso.python.token import PythonTokenTypes
from parso.parser import BaseParser
def current_suite(stack):
    for until_index, stack_node in reversed(list(enumerate(stack))):
        if stack_node.nonterminal == 'file_input':
            break
        elif stack_node.nonterminal == 'suite':
            if len(stack_node.nodes) != 1:
                break
    return until_index