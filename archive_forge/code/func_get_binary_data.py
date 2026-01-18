import uuid
import numpy as np
from ..data_utils import is_pandas_df, has_geo_interface, records_from_geo_interface
from .json_tools import JSONMixin, camel_and_lower
from ..settings import settings as pydeck_settings
from pydeck.types import Image, Function
from pydeck.exceptions import BinaryTransportException
def get_binary_data(self):
    if not self.use_binary_transport:
        raise BinaryTransportException('Layer must be flagged with `use_binary_transport=True`')
    return self._binary_data