from nbconvert.preprocessors.clearoutput import ClearOutputPreprocessor
from nbclient.exceptions import CellExecutionError
from nbclient.client import NotebookClient
from traitlets import Bool, Unicode
def strip_code_cell_errors(self, cell):
    """Strip any error outputs and traceback from a code cell."""
    if cell['cell_type'] != 'code':
        return cell
    outputs = cell['outputs']
    error_outputs = [output for output in outputs if output['output_type'] == 'error']
    error_message = 'There was an error when executing cell [{}]. {}'.format(cell['execution_count'], self.cell_error_instruction)
    for output in error_outputs:
        output['ename'] = 'ExecutionError'
        output['evalue'] = 'Execution error'
        output['traceback'] = [error_message]
    return cell