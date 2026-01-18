import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
def check_data_descriptor(mock_attr):
    self.assertIsInstance(mock_attr, MagicMock)
    mock_attr(1, 2, 3)
    mock_attr.abc(4, 5, 6)
    mock_attr.assert_called_once_with(1, 2, 3)
    mock_attr.abc.assert_called_once_with(4, 5, 6)