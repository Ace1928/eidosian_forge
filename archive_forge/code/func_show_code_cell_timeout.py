from nbconvert.preprocessors.clearoutput import ClearOutputPreprocessor
from nbclient.exceptions import CellExecutionError
from nbclient.client import NotebookClient
from traitlets import Bool, Unicode
def show_code_cell_timeout(self, cell):
    """Show a timeout error output in a code cell."""
    timeout_message = 'Cell execution timed out, aborting notebook execution. {}'.format(self.cell_timeout_instruction)
    output = {'output_type': 'error', 'ename': 'TimeoutError', 'evalue': 'Timeout error', 'traceback': [timeout_message]}
    cell['outputs'] = [output]