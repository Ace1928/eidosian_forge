from nbconvert.preprocessors.clearoutput import ClearOutputPreprocessor
from nbclient.exceptions import CellExecutionError
from nbclient.client import NotebookClient
from traitlets import Bool, Unicode
def should_strip_error(self):
    """Return True if errors should be stripped from the Notebook, False otherwise, depending on the current config."""
    return not self.show_tracebacks