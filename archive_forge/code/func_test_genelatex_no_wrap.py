from contextlib import contextmanager
from unittest.mock import patch
import pytest
from IPython.lib import latextools
from IPython.testing.decorators import (
from IPython.utils.process import FindCmdError
def test_genelatex_no_wrap():
    """
    Test genelatex with wrap=False.
    """

    def mock_kpsewhich(filename):
        assert False, 'kpsewhich should not be called (called with {0})'.format(filename)
    with patch_latextool(mock_kpsewhich):
        assert '\n'.join(latextools.genelatex('body text', False)) == '\\documentclass{article}\n\\usepackage{amsmath}\n\\usepackage{amsthm}\n\\usepackage{amssymb}\n\\usepackage{bm}\n\\pagestyle{empty}\n\\begin{document}\nbody text\n\\end{document}'