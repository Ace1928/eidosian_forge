import datetime as dt
from unittest import TestCase
from unittest.mock import MagicMock
from traitlets import TraitError
from ipywidgets import FileUpload
def test_empty_initial_value(self):
    uploader = FileUpload()
    assert uploader.value == ()