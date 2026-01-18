import gc
import weakref
import pytest
def handle_test_exception(e):
    nonlocal exception
    exception = str(e)