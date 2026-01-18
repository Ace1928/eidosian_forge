import inspect
from collections.abc import Iterable
from typing import Optional, Text
@staticmethod
def set_input_argument(input_name: Text='inputs') -> None:
    """
        Set the name of the input argument.

        Args:
            input_name (str): Name of the input argument
        """
    KerasLayer._input_arg = input_name