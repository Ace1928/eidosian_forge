from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, TypeVar, cast
from pymongo.server_type import SERVER_TYPE
def secondary_with_tags_server_selector(tag_sets: TagSets, selection: Selection) -> Selection:
    """All near-enough secondaries matching the tag sets."""
    return apply_tag_sets(tag_sets, secondary_server_selector(selection))