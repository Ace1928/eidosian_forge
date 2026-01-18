import re
from typing import Any, Callable, Optional
from lxml import etree
from w3lib.html import HTML5_WHITESPACE
def has_class(context: Any, *classes: str) -> bool:
    """has-class function.

    Return True if all ``classes`` are present in element's class attr.

    """
    if not context.eval_context.get('args_checked'):
        if not classes:
            raise ValueError('XPath error: has-class must have at least 1 argument')
        for c in classes:
            if not isinstance(c, str):
                raise ValueError('XPath error: has-class arguments must be strings')
        context.eval_context['args_checked'] = True
    node_cls = context.context_node.get('class')
    if node_cls is None:
        return False
    node_cls = ' ' + node_cls + ' '
    node_cls = replace_html5_whitespaces(' ', node_cls)
    for cls in classes:
        if ' ' + cls + ' ' not in node_cls:
            return False
    return True