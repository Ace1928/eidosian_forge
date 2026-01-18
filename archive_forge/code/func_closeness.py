from __future__ import absolute_import
import ast
from sentry_sdk import Hub, serializer
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.scope import add_global_event_processor
from sentry_sdk.utils import walk_exception_chain, iter_stacks
def closeness(expression):
    nodes, _value = expression

    def start(n):
        return (n.lineno, n.col_offset)
    nodes_before_stmt = [node for node in nodes if start(node) < stmt.last_token.end]
    if nodes_before_stmt:
        return max((start(node) for node in nodes_before_stmt))
    else:
        lineno, col_offset = min((start(node) for node in nodes))
        return (-lineno, -col_offset)