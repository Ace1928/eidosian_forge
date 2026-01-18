from contextlib import contextmanager
from unittest.mock import patch
import pytest
from IPython.lib import latextools
from IPython.testing.decorators import (
from IPython.utils.process import FindCmdError
def test_genelatex_wrap_with_breqn():
    """
    Test genelatex with wrap=True for the case breqn.sty is installed.
    """

    def mock_kpsewhich(filename):
        assert filename == 'breqn.sty'
        return 'path/to/breqn.sty'
    with patch_latextool(mock_kpsewhich):
        assert '\n'.join(latextools.genelatex('x^2', True)) == '\\documentclass{article}\n\\usepackage{amsmath}\n\\usepackage{amsthm}\n\\usepackage{amssymb}\n\\usepackage{bm}\n\\usepackage{breqn}\n\\pagestyle{empty}\n\\begin{document}\n\\begin{dmath*}\nx^2\n\\end{dmath*}\n\\end{document}'