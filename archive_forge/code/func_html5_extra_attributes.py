from functools import reduce
from datetime import datetime
import re
def html5_extra_attributes(node, state):
    """
    @param node: the current node that could be modified
    @param state: current state
    @type state: L{Execution context<pyRdfa.state.ExecutionContext>}
    """

    def _get_literal(Pnode):
        """
        Get (recursively) the full text from a DOM Node.
    
        @param Pnode: DOM Node
        @return: string
        """
        rc = ''
        for node in Pnode.childNodes:
            if node.nodeType == node.TEXT_NODE:
                rc = rc + node.data
            elif node.nodeType == node.ELEMENT_NODE:
                rc = rc + _get_literal(node)
        if state.options.space_preserve:
            return rc
        else:
            return re.sub('(\\r| |\\n|\\t)+', ' ', rc).strip()

    def _set_time(value):
        if not node.hasAttribute('datatype'):
            dt = _format_test(value)
            if dt != plain:
                node.setAttribute('datatype', dt)
        node.setAttribute('content', value)
    if not node.hasAttribute('content'):
        if node.hasAttribute('datetime'):
            _set_time(node.getAttribute('datetime'))
        elif node.hasAttribute('dateTime'):
            _set_time(node.getAttribute('dateTime'))
        elif node.tagName == 'time':
            _set_time(_get_literal(node))