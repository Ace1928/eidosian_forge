import re
from io import BytesIO
from .. import errors
def read_pending_records(self, max=None):
    if max:
        records = self._parsed_records[:max]
        del self._parsed_records[:max]
        return records
    else:
        records = self._parsed_records
        self._parsed_records = []
        return records