from __future__ import annotations
import copy
import typing as t
from . import common
def start_ya_feed(self, attrs: dict[str, str]) -> None:
    if attrs.get(RDF_RESOURCE, '').strip():
        agent_feed = attrs[RDF_RESOURCE].strip()
        self.agent_feeds.append(agent_feed)