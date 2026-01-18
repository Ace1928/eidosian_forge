import copy
import warnings
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.pulse.schedule import Schedule
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states import statevector
from qiskit.result.models import ExperimentResult
from qiskit.result import postprocess
from qiskit.result.counts import Counts
from qiskit.qobj.utils import MeasLevel
from qiskit.qobj import QobjHeader
def get_statevector(self, experiment=None, decimals=None):
    """Get the final statevector of an experiment.

        Args:
            experiment (str or QuantumCircuit or Schedule or int or None): the index of the
                experiment, as specified by ``data()``.
            decimals (int): the number of decimals in the statevector.
                If None, does not round.

        Returns:
            list[complex]: list of 2^num_qubits complex amplitudes.

        Raises:
            QiskitError: if there is no statevector for the experiment.
        """
    try:
        return postprocess.format_statevector(self.data(experiment)['statevector'], decimals=decimals)
    except KeyError as ex:
        raise QiskitError(f'No statevector for experiment "{repr(experiment)}"') from ex