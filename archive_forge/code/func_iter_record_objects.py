import re
from io import BytesIO
from .. import errors
def iter_record_objects(self):
    """Iterate over the container, yielding each record as it is read.

        Each yielded record will be an object with ``read`` and ``validate``
        methods.  Like with iter_records, it is not safe to use a record object
        after advancing the iterator to yield next record.

        :raises ContainerError: if any sort of container corruption is
            detected, e.g. UnknownContainerFormatError is the format of the
            container is unrecognised.
        :seealso: iter_records
        """
    self._read_format()
    return self._iter_record_objects()