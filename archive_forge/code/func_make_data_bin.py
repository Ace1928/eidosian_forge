from __future__ import annotations
from collections.abc import Iterable
from dataclasses import make_dataclass
def make_data_bin(fields: Iterable[tuple[str, type]], shape: tuple[int, ...] | None=None) -> DataBinMeta:
    """Return a new subclass of :class:`~DataBin` with the provided fields and shape.

    .. code-block:: python

        my_bin = make_data_bin([("alpha", np.NDArray[np.float64])], shape=(20, 30))

        # behaves like a dataclass
        my_bin(alpha=np.empty((20, 30)))

    Args:
        fields: Tuples ``(name, type)`` specifying the attributes of the returned class.
        shape: The intended shape of every attribute of this class.

    Returns:
        A new class.
    """
    field_names, field_types = zip(*fields) if fields else ([], [])
    for name in field_names:
        if name in DataBin._RESTRICTED_NAMES:
            raise ValueError(f"'{name}' is a restricted name for a DataBin.")
    cls = make_dataclass('DataBin', dict(zip(field_names, field_types)), bases=(DataBin,), frozen=True, unsafe_hash=True, repr=False)
    cls._SHAPE = shape
    cls._FIELDS = field_names
    cls._FIELD_TYPES = field_types
    return cls