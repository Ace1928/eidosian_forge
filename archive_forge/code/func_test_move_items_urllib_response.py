import operator
import sys
import types
import unittest
import abc
import pytest
import six
@pytest.mark.parametrize('item_name', [item.name for item in six._urllib_response_moved_attributes])
def test_move_items_urllib_response(item_name):
    """Ensure that everything loads correctly."""
    assert item_name in dir(six.moves.urllib.response)
    getattr(six.moves.urllib.response, item_name)