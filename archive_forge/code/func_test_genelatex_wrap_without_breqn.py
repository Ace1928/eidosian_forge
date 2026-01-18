from contextlib import contextmanager
from unittest.mock import patch
import pytest
from IPython.lib import latextools
from IPython.testing.decorators import (
from IPython.utils.process import FindCmdError
def test_genelatex_wrap_without_breqn():
    """
    Test genelatex with wrap=True for the case breqn.sty is not installed.
    """

    def mock_kpsewhich(filename):
        assert filename == 'breqn.sty'
        return None
    with patch_latextool(mock_kpsewhich):
        assert '\n'.join(latextools.genelatex('x^2', True)) == '\\documentclass{article}\n\\usepackage{amsmath}\n\\usepackage{amsthm}\n\\usepackage{amssymb}\n\\usepackage{bm}\n\\pagestyle{empty}\n\\begin{document}\n$$x^2$$\n\\end{document}'