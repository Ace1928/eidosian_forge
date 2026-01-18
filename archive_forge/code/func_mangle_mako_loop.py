import json
import re
import time
from mako import ast
from mako import exceptions
from mako import filters
from mako import parsetree
from mako import util
from mako.pygen import PythonPrinter
def mangle_mako_loop(node, printer):
    """converts a for loop into a context manager wrapped around a for loop
    when access to the `loop` variable has been detected in the for loop body
    """
    loop_variable = LoopVariable()
    node.accept_visitor(loop_variable)
    if loop_variable.detected:
        node.nodes[-1].has_loop_context = True
        match = _FOR_LOOP.match(node.text)
        if match:
            printer.writelines('loop = __M_loop._enter(%s)' % match.group(2), 'try:')
            text = 'for %s in loop:' % match.group(1)
        else:
            raise SyntaxError("Couldn't apply loop context: %s" % node.text)
    else:
        text = node.text
    return text