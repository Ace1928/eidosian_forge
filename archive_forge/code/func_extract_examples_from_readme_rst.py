import contextlib
import inspect
import io
import os.path
import re
from . import compression
from . import transport
def extract_examples_from_readme_rst(indent='    '):
    """Extract examples from this project's README.rst file.

    Parameters
    ----------
    indent: str
        Prepend each line with this string.  Should contain some number of spaces.

    Returns
    -------
    str
        The examples.

    Notes
    -----
    Quite fragile, depends on named labels inside the README.rst file.
    """
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    readme_path = os.path.join(curr_dir, '..', 'README.rst')
    try:
        with open(readme_path) as fin:
            lines = list(fin)
        start = lines.index('.. _doctools_before_examples:\n')
        end = lines.index('.. _doctools_after_examples:\n')
        lines = lines[start + 4:end - 2]
        return ''.join([indent + re.sub('^  ', '', line) for line in lines])
    except Exception:
        return indent + 'See README.rst'