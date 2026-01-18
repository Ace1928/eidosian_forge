from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import SSHKey
def get_by_fingerprint(self, fingerprint: str) -> BoundSSHKey | None:
    """Get ssh key by fingerprint

        :param fingerprint: str
                Used to get ssh key by fingerprint.
        :return: :class:`BoundSSHKey <hcloud.ssh_keys.client.BoundSSHKey>`
        """
    return self._get_first_by(fingerprint=fingerprint)