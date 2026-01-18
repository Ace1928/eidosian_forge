import os
import sys
from pathlib import Path
import pytest
import srsly
from typer.testing import CliRunner
from spacy.cli._util import app, get_git_version
from spacy.tokens import Doc, DocBin, Span
from .util import make_tempdir, normalize_whitespace
def test_find_function_valid():
    function = 'spacy.TextCatBOW.v3'
    result = CliRunner().invoke(app, ['find-function', function, '-r', 'architectures'])
    assert f"Found registered function '{function}'" in result.stdout
    assert 'textcat.py' in result.stdout
    result = CliRunner().invoke(app, ['find-function', function])
    assert f"Found registered function '{function}'" in result.stdout
    assert 'textcat.py' in result.stdout
    function = 'spacy.TextCatBOW.v1'
    result = CliRunner().invoke(app, ['find-function', function])
    assert f"Found registered function '{function}'" in result.stdout
    assert 'spacy_legacy' in result.stdout
    assert 'textcat.py' in result.stdout