from __future__ import annotations
import copy
import typing as t
from . import common
def start_foaf_document(self, attrs: dict[str, str]) -> None:
    if attrs.get(RDF_ABOUT, '').strip():
        self.flag_opportunity = True
        agent_opp = attrs[RDF_ABOUT].strip()
        self.agent_opps.append(agent_opp)