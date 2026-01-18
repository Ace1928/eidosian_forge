from __future__ import annotations
from typing import TYPE_CHECKING, Any, Tuple, Type, Union
from uuid import UUID
@classmethod
def from_uuid(cls: Type[Binary], uuid: UUID, uuid_representation: int=UuidRepresentation.STANDARD) -> Binary:
    """Create a BSON Binary object from a Python UUID.

        Creates a :class:`~bson.binary.Binary` object from a
        :class:`uuid.UUID` instance. Assumes that the native
        :class:`uuid.UUID` instance uses the byte-order implied by the
        provided ``uuid_representation``.

        Raises :exc:`TypeError` if `uuid` is not an instance of
        :class:`~uuid.UUID`.

        :Parameters:
          - `uuid`: A :class:`uuid.UUID` instance.
          - `uuid_representation`: A member of
            :class:`~bson.binary.UuidRepresentation`. Default:
            :const:`~bson.binary.UuidRepresentation.STANDARD`.
            See :ref:`handling-uuid-data-example` for details.

        .. versionadded:: 3.11
        """
    if not isinstance(uuid, UUID):
        raise TypeError('uuid must be an instance of uuid.UUID')
    if uuid_representation not in ALL_UUID_REPRESENTATIONS:
        raise ValueError('uuid_representation must be a value from bson.binary.UuidRepresentation')
    if uuid_representation == UuidRepresentation.UNSPECIFIED:
        raise ValueError('cannot encode native uuid.UUID with UuidRepresentation.UNSPECIFIED. UUIDs can be manually converted to bson.Binary instances using bson.Binary.from_uuid() or a different UuidRepresentation can be configured. See the documentation for UuidRepresentation for more information.')
    subtype = OLD_UUID_SUBTYPE
    if uuid_representation == UuidRepresentation.PYTHON_LEGACY:
        payload = uuid.bytes
    elif uuid_representation == UuidRepresentation.JAVA_LEGACY:
        from_uuid = uuid.bytes
        payload = from_uuid[0:8][::-1] + from_uuid[8:16][::-1]
    elif uuid_representation == UuidRepresentation.CSHARP_LEGACY:
        payload = uuid.bytes_le
    else:
        subtype = UUID_SUBTYPE
        payload = uuid.bytes
    return cls(payload, subtype)