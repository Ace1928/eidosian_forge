import pytest
from unittest import mock
from traitlets import Bool, Tuple, List, Instance, CFloat, CInt, Float, Int, TraitError, observe
from .utils import setup, teardown
import ipywidgets
from ipywidgets import Widget
def bytes_serializer(instance, widget):
    return {'data': bytearray(memoryview(instance.data).tobytes()) if instance.data else None}