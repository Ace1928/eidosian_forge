import os
import sys
from pathlib import Path
import pytest
import srsly
from typer.testing import CliRunner
from spacy.cli._util import app, get_git_version
from spacy.tokens import Doc, DocBin, Span
from .util import make_tempdir, normalize_whitespace
def test_find_function_invalid():
    function = 'spacy.TextCatBOW.v3'
    registry = 'foobar'
    result = CliRunner().invoke(app, ['find-function', function, '--registry', registry])
    assert f"Unknown function registry: '{registry}'" in result.stdout
    function = 'spacy.TextCatBOW.v666'
    result = CliRunner().invoke(app, ['find-function', function])
    assert f"Couldn't find registered function: '{function}'" in result.stdout