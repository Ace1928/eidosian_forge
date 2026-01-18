import os
import sys
from pathlib import Path
import pytest
import srsly
from typer.testing import CliRunner
from spacy.cli._util import app, get_git_version
from spacy.tokens import Doc, DocBin, Span
from .util import make_tempdir, normalize_whitespace
def test_convert_auto_conflict():
    with make_tempdir() as d_in, make_tempdir() as d_out:
        for f in ['data1.iob', 'data2.iob', 'data3.json']:
            Path(d_in / f).touch()
        result = CliRunner().invoke(app, ['convert', str(d_in), str(d_out)])
        assert 'All input files must be same type' in result.stdout
        out_files = os.listdir(d_out)
        assert len(out_files) == 0