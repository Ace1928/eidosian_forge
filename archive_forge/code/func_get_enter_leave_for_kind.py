from copy import copy
from enum import Enum
from typing import (
from ..pyutils import inspect, snake_to_camel
from . import ast
from .ast import Node, QUERY_DOCUMENT_KEYS
def get_enter_leave_for_kind(self, kind: str) -> EnterLeaveVisitor:
    """Given a node kind, return the EnterLeaveVisitor for that kind."""
    try:
        return self.enter_leave_map[kind]
    except KeyError:
        has_visitor = False
        enter_list: List[Optional[Callable[..., Optional[VisitorAction]]]] = []
        leave_list: List[Optional[Callable[..., Optional[VisitorAction]]]] = []
        for visitor in self.visitors:
            enter, leave = visitor.get_enter_leave_for_kind(kind)
            if not has_visitor and (enter or leave):
                has_visitor = True
            enter_list.append(enter)
            leave_list.append(leave)
        if has_visitor:

            def enter(node: Node, *args: Any) -> Optional[VisitorAction]:
                skipping = self.skipping
                for i, fn in enumerate(enter_list):
                    if not skipping[i]:
                        if fn:
                            result = fn(node, *args)
                            if result is SKIP or result is False:
                                skipping[i] = node
                            elif result is BREAK or result is True:
                                skipping[i] = BREAK
                            elif result is not None:
                                return result
                return None

            def leave(node: Node, *args: Any) -> Optional[VisitorAction]:
                skipping = self.skipping
                for i, fn in enumerate(leave_list):
                    if not skipping[i]:
                        if fn:
                            result = fn(node, *args)
                            if result is BREAK or result is True:
                                skipping[i] = BREAK
                            elif result is not None and result is not SKIP and (result is not False):
                                return result
                    elif skipping[i] is node:
                        skipping[i] = None
                return None
        else:
            enter = leave = None
        enter_leave = EnterLeaveVisitor(enter, leave)
        self.enter_leave_map[kind] = enter_leave
        return enter_leave