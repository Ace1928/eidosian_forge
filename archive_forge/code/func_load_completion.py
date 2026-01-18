from __future__ import annotations
import abc
import dataclasses
import enum
import os
import typing as t
from .constants import (
from .util import (
from .data import (
from .become import (
def load_completion(name: str, completion_type: t.Type[TCompletionConfig]) -> dict[str, TCompletionConfig]:
    """Load the named completion entries, returning them in dictionary form using the specified completion type."""
    lines = read_lines_without_comments(os.path.join(ANSIBLE_TEST_DATA_ROOT, 'completion', '%s.txt' % name), remove_blank_lines=True)
    if data_context().content.collection:
        context = 'collection'
    else:
        context = 'ansible-core'
    items = {name: data for name, data in [parse_completion_entry(line) for line in lines] if data.get('context', context) == context}
    for item in items.values():
        item.pop('context', None)
        item.pop('placeholder', None)
    completion = {name: completion_type(name=name, **data) for name, data in items.items()}
    return completion