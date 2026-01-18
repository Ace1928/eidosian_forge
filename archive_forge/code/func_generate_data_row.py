import copy
import io
from collections import (
from enum import (
from typing import (
from wcwidth import (  # type: ignore[import]
from . import (
def generate_data_row(self, row_data: Sequence[Any]) -> str:
    """
        Generate a data row

        :param row_data: data with an entry for each column in the row
        :return: data row string
        """
    row = super().generate_data_row(row_data)
    self.row_num += 1
    return row