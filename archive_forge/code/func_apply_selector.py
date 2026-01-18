from __future__ import annotations
from random import sample
from typing import (
from bson.min_key import MinKey
from bson.objectid import ObjectId
from pymongo import common
from pymongo.errors import ConfigurationError
from pymongo.read_preferences import ReadPreference, _AggWritePref, _ServerMode
from pymongo.server_description import ServerDescription
from pymongo.server_selectors import Selection
from pymongo.server_type import SERVER_TYPE
from pymongo.typings import _Address
def apply_selector(self, selector: Any, address: Optional[_Address]=None, custom_selector: Optional[_ServerSelector]=None) -> list[ServerDescription]:
    """List of servers matching the provided selector(s).

        :Parameters:
          - `selector`: a callable that takes a Selection as input and returns
            a Selection as output. For example, an instance of a read
            preference from :mod:`~pymongo.read_preferences`.
          - `address` (optional): A server address to select.
          - `custom_selector` (optional): A callable that augments server
            selection rules. Accepts a list of
            :class:`~pymongo.server_description.ServerDescription` objects and
            return a list of server descriptions that should be considered
            suitable for the desired operation.

        .. versionadded:: 3.4
        """
    if getattr(selector, 'min_wire_version', 0):
        common_wv = self.common_wire_version
        if common_wv and common_wv < selector.min_wire_version:
            raise ConfigurationError("%s requires min wire version %d, but topology's min wire version is %d" % (selector, selector.min_wire_version, common_wv))
    if isinstance(selector, _AggWritePref):
        selector.selection_hook(self)
    if self.topology_type == TOPOLOGY_TYPE.Unknown:
        return []
    elif self.topology_type in (TOPOLOGY_TYPE.Single, TOPOLOGY_TYPE.LoadBalanced):
        return self.known_servers
    if address:
        description = self.server_descriptions().get(address)
        return [description] if description else []
    selection = Selection.from_topology_description(self)
    if self.topology_type != TOPOLOGY_TYPE.Sharded:
        selection = selector(selection)
    if custom_selector is not None and selection:
        selection = selection.with_server_descriptions(custom_selector(selection.server_descriptions))
    return self._apply_local_threshold(selection)