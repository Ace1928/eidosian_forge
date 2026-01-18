from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, TypeVar, cast
from pymongo.server_type import SERVER_TYPE
def apply_tag_sets(tag_sets: TagSets, selection: Selection) -> Selection:
    """All servers match a list of tag sets.

    tag_sets is a list of dicts. The empty tag set {} matches any server,
    and may be provided at the end of the list as a fallback. So
    [{'a': 'value'}, {}] expresses a preference for servers tagged
    {'a': 'value'}, but accepts any server if none matches the first
    preference.
    """
    for tag_set in tag_sets:
        with_tag_set = apply_single_tag_set(tag_set, selection)
        if with_tag_set:
            return with_tag_set
    return selection.with_server_descriptions([])