from pathlib import Path
from typing import Iterable
import cirq
import cirq.contrib.svg.svg as ccsvg
import cirq_ft.infra.testing as cq_testing
import IPython.display
import ipywidgets
import nbformat
from cirq_ft.infra import gate_with_registers, t_complexity_protocol, get_named_qubits, merge_qubits
from nbconvert.preprocessors import ExecutePreprocessor
def execute_notebook(name: str):
    """Execute a jupyter notebook in the caller's directory.

    Args:
        name: The name of the notebook without extension.
    """
    import traceback
    notebook_path = Path(traceback.extract_stack()[-2].filename).parent / f'{name}.ipynb'
    with notebook_path.open() as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb)