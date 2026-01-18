import pytest
from textwrap import dedent
import platform
import srsly
from .roundtrip import (
def test_anchor_on_sequence(self):
    from srsly.ruamel_yaml.comments import CommentedSeq
    data = load('\n        nut1: &alice\n         - 1\n         - 2\n        nut2: &blake\n         - some data\n         - *alice\n        nut3:\n         - *blake\n         - *alice\n        ')
    r = data['nut1']
    assert isinstance(r, CommentedSeq)
    assert r.yaml_anchor() is not None
    assert r.yaml_anchor().value == 'alice'