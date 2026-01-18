from __future__ import annotations
import copy
import typing as t
from . import common
def start_rss_channel(self, attrs: dict[str, str]) -> None:
    if attrs.get(RDF_ABOUT, '').strip():
        if self.flag_opportunity:
            self.flag_opportunity = False
            self.agent_opps.pop()
        agent_feed = attrs[RDF_ABOUT].strip()
        self.agent_feeds.append(agent_feed)