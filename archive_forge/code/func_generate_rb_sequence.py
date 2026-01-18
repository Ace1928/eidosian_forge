from typing import List, Optional, Sequence, cast
from qcs_api_client.client import QCSClientConfiguration
from pyquil.api._abstract_compiler import AbstractBenchmarker
from pyquil.api._compiler_client import (
from pyquil.paulis import PauliTerm, is_identity
from pyquil.quil import address_qubits, Program
from pyquil.quilbase import Gate
def generate_rb_sequence(self, depth: int, gateset: Sequence[Gate], seed: Optional[int]=None, interleaver: Optional[Program]=None) -> List[Program]:
    """
        Construct a randomized benchmarking experiment on the given qubits, decomposing into
        gateset. If interleaver is not provided, the returned sequence will have the form

            C_1 C_2 ... C_(depth-1) C_inv ,

        where each C is a Clifford element drawn from gateset, C_{< depth} are randomly selected,
        and C_inv is selected so that the entire sequence composes to the identity.  If an
        interleaver G (which must be a Clifford, and which will be decomposed into the native
        gateset) is provided, then the sequence instead takes the form

            C_1 G C_2 G ... C_(depth-1) G C_inv .

        The JSON response is a list of lists of indices, or Nones. In the former case, they are the
        index of the gate in the gateset.

        :param depth: The number of Clifford gates to include in the randomized benchmarking
         experiment. This is different than the number of gates in the resulting experiment.
        :param gateset: A list of pyquil gates to decompose the Clifford elements into. These
         must generate the clifford group on the qubits of interest. e.g. for one qubit
         [RZ(np.pi/2), RX(np.pi/2)].
        :param seed: A positive integer used to seed the PRNG.
        :param interleaver: A Program object that encodes a Clifford element.
        :return: A list of pyquil programs. Each pyquil program is a circuit that represents an
         element of the Clifford group. When these programs are composed, the resulting Program
         will be the randomized benchmarking experiment of the desired depth. e.g. if the return
         programs are called cliffords then `sum(cliffords, Program())` will give the randomized
         benchmarking experiment, which will compose to the identity program.
        """
    gateset_as_program = address_qubits(sum(gateset, Program()))
    qubits = len(gateset_as_program.get_qubits())
    gateset_for_api = gateset_as_program.out().splitlines()
    interleaver_out: Optional[str] = None
    if interleaver:
        assert isinstance(interleaver, Program)
        interleaver_out = interleaver.out(calibrations=False)
    depth = int(depth)
    request = GenerateRandomizedBenchmarkingSequenceRequest(depth=depth, num_qubits=qubits, gateset=gateset_for_api, seed=seed, interleaver=interleaver_out)
    response = self._compiler_client.generate_randomized_benchmarking_sequence(request)
    programs = []
    for clifford in response.sequence:
        clifford_program = Program()
        if interleaver:
            clifford_program._calibrations = interleaver.calibrations
        for index in reversed(clifford):
            clifford_program.inst(gateset[index])
        programs.append(clifford_program)
    return list(reversed(programs))