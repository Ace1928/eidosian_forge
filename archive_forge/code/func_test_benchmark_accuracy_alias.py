import os
import sys
from pathlib import Path
import pytest
import srsly
from typer.testing import CliRunner
from spacy.cli._util import app, get_git_version
from spacy.tokens import Doc, DocBin, Span
from .util import make_tempdir, normalize_whitespace
def test_benchmark_accuracy_alias():
    result_benchmark = CliRunner().invoke(app, ['benchmark', 'accuracy', '--help'])
    result_evaluate = CliRunner().invoke(app, ['evaluate', '--help'])
    assert normalize_whitespace(result_benchmark.stdout) == normalize_whitespace(result_evaluate.stdout.replace('spacy evaluate', 'spacy benchmark accuracy'))