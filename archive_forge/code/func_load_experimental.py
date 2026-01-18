import functools
import warnings
from qiskit import _qasm3
from qiskit.exceptions import ExperimentalWarning
from qiskit.utils import optionals as _optionals
from .experimental import ExperimentalFeatures
from .exporter import Exporter
from .exceptions import QASM3Error, QASM3ImporterError, QASM3ExporterError
from .._qasm3 import CustomGate, STDGATES_INC_GATES
@functools.wraps(_qasm3.load)
def load_experimental(pathlike_or_filelike, /, *, custom_gates=None, include_path=None):
    """<overridden by functools.wraps>"""
    warnings.warn('This is an experimental native version of the OpenQASM 3 importer. Beware that its interface might change, and it might be missing features.', category=ExperimentalWarning)
    return _qasm3.load(pathlike_or_filelike, custom_gates=custom_gates, include_path=include_path)