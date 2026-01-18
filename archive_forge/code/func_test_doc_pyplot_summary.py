import difflib
import numpy as np
import sys
from pathlib import Path
import pytest
import matplotlib as mpl
from matplotlib.testing import subprocess_run_for_testing
from matplotlib import pyplot as plt
def test_doc_pyplot_summary():
    """Test that pyplot_summary lists all the plot functions."""
    pyplot_docs = Path(__file__).parent / '../../../doc/api/pyplot_summary.rst'
    if not pyplot_docs.exists():
        pytest.skip('Documentation sources not available')

    def extract_documented_functions(lines):
        """
        Return a list of all the functions that are mentioned in the
        autosummary blocks contained in *lines*.

        An autosummary block looks like this::

            .. autosummary::
               :toctree: _as_gen
               :template: autosummary.rst
               :nosignatures:

               plot
               plot_date

        """
        functions = []
        in_autosummary = False
        for line in lines:
            if not in_autosummary:
                if line.startswith('.. autosummary::'):
                    in_autosummary = True
            else:
                if not line or line.startswith('   :'):
                    continue
                if not line[0].isspace():
                    in_autosummary = False
                    continue
                functions.append(line.strip())
        return functions
    lines = pyplot_docs.read_text().split('\n')
    doc_functions = set(extract_documented_functions(lines))
    plot_commands = set(plt._get_pyplot_commands())
    missing = plot_commands.difference(doc_functions)
    if missing:
        raise AssertionError(f'The following pyplot functions are not listed in the documentation. Please add them to doc/api/pyplot_summary.rst: {missing!r}')
    extra = doc_functions.difference(plot_commands)
    if extra:
        raise AssertionError(f'The following functions are listed in the pyplot documentation, but they do not exist in pyplot. Please remove them from doc/api/pyplot_summary.rst: {extra!r}')