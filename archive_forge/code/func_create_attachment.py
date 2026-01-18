import typing as ty
from openstack.block_storage import _base_proxy
from openstack.block_storage.v3 import attachment as _attachment
from openstack.block_storage.v3 import availability_zone
from openstack.block_storage.v3 import backup as _backup
from openstack.block_storage.v3 import block_storage_summary as _summary
from openstack.block_storage.v3 import capabilities as _capabilities
from openstack.block_storage.v3 import extension as _extension
from openstack.block_storage.v3 import group as _group
from openstack.block_storage.v3 import group_snapshot as _group_snapshot
from openstack.block_storage.v3 import group_type as _group_type
from openstack.block_storage.v3 import limits as _limits
from openstack.block_storage.v3 import quota_set as _quota_set
from openstack.block_storage.v3 import resource_filter as _resource_filter
from openstack.block_storage.v3 import service as _service
from openstack.block_storage.v3 import snapshot as _snapshot
from openstack.block_storage.v3 import stats as _stats
from openstack.block_storage.v3 import transfer as _transfer
from openstack.block_storage.v3 import type as _type
from openstack.block_storage.v3 import volume as _volume
from openstack import exceptions
from openstack.identity.v3 import project as _project
from openstack import resource
from openstack import utils
def create_attachment(self, volume, **attrs):
    """Create a new attachment

        This is an internal API and should only be called by services
        consuming volume attachments like nova, glance, ironic etc.

        :param volume: The value can be either the ID of a volume or a
            :class:`~openstack.block_storage.v3.volume.Volume` instance.
        :param dict attrs: Keyword arguments which will be used to create
            a :class:`~openstack.block_storage.v3.attachment.Attachment`
            comprised of the properties on the Attachment class like
            connector, instance_id, mode etc.
        :returns: The results of attachment creation
        :rtype: :class:`~openstack.block_storage.v3.attachment.Attachment`
        """
    volume_id = resource.Resource._get_id(volume)
    return self._create(_attachment.Attachment, volume_id=volume_id, **attrs)