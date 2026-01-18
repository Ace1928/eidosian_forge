from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import ActionsPageResult, BoundAction, ResourceActionsClient
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import Image
def get_by_name_and_architecture(self, name: str, architecture: str) -> BoundImage | None:
    """Get image by name

        :param name: str
               Used to identify the image.
        :param architecture: str
               Used to identify the image.
        :return: :class:`BoundImage <hcloud.images.client.BoundImage>`
        """
    return self._get_first_by(name=name, architecture=[architecture])