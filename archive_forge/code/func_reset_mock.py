from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def reset_mock(self, visited=None):
    """Restore the mock object to its initial state."""
    if visited is None:
        visited = []
    if id(self) in visited:
        return
    visited.append(id(self))
    self.called = False
    self.call_args = None
    self.call_count = 0
    self.mock_calls = _CallList()
    self.call_args_list = _CallList()
    self.method_calls = _CallList()
    for child in self._mock_children.values():
        if isinstance(child, _SpecState):
            continue
        child.reset_mock(visited)
    ret = self._mock_return_value
    if _is_instance_mock(ret) and ret is not self:
        ret.reset_mock(visited)