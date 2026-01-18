from __future__ import annotations
import copy
import typing as t
from . import common
def start_rdfs_seealso(self, attrs: dict[str, str]) -> None:
    if attrs.get(RDF_RESOURCE, '').strip():
        agent_list = attrs[RDF_RESOURCE].strip()
        self.agent_lists.append(agent_list)