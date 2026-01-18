import itertools
import types
import warnings
from collections import defaultdict
from typing import (
import numpy as np
from rpcq.messages import NativeQuilMetadata, ParameterAref
from pyquil._parser.parser import run_parser
from pyquil._memory import Memory
from pyquil.gates import MEASURE, RESET, MOVE
from pyquil.noise import _check_kraus_ops, _create_kraus_pragmas, pauli_kraus_map
from pyquil.quilatom import (
from pyquil.quilbase import (
from pyquil.quiltcalibrations import (
def inst(self, *instructions: InstructionDesignator) -> 'Program':
    """
        Mutates the Program object by appending new instructions.

        This function accepts a number of different valid forms, e.g.

            >>> p = Program()
            >>> p.inst(H(0)) # A single instruction
            >>> p.inst(H(0), H(1)) # Multiple instructions
            >>> p.inst([H(0), H(1)]) # A list of instructions
            >>> p.inst(H(i) for i in range(4)) # A generator of instructions
            >>> p.inst(("H", 1)) # A tuple representing an instruction
            >>> p.inst("H 0") # A string representing an instruction
            >>> q = Program()
            >>> p.inst(q) # Another program

        It can also be chained:
            >>> p = Program()
            >>> p.inst(H(0)).inst(H(1))

        :param instructions: A list of Instruction objects, e.g. Gates
        :return: self for method chaining
        """
    for instruction in instructions:
        if isinstance(instruction, list):
            self.inst(*instruction)
        elif isinstance(instruction, types.GeneratorType):
            self.inst(*instruction)
        elif isinstance(instruction, tuple):
            if len(instruction) == 0:
                raise ValueError('tuple should have at least one element')
            elif len(instruction) == 1:
                self.inst(instruction[0])
            else:
                op = instruction[0]
                if op == 'MEASURE':
                    if len(instruction) == 2:
                        self.measure(instruction[1], None)
                    else:
                        self.measure(instruction[1], instruction[2])
                else:
                    params: List[ParameterDesignator] = []
                    possible_params = instruction[1]
                    rest: Sequence[Any] = instruction[2:]
                    if isinstance(possible_params, list):
                        params = possible_params
                    else:
                        rest = [possible_params] + list(rest)
                    self.gate(op, params, rest)
        elif isinstance(instruction, str):
            self.inst(run_parser(instruction.strip()))
        elif isinstance(instruction, Program):
            if id(self) == id(instruction):
                raise ValueError('Nesting a program inside itself is not supported')
            for defgate in instruction._defined_gates:
                self.inst(defgate)
            for instr in instruction._instructions:
                self.inst(instr)
        elif isinstance(instruction, DefGate):
            r_idx, existing_defgate = next(((i, gate) for i, gate in enumerate(reversed(self._defined_gates)) if gate.name == instruction.name), (0, None))
            if existing_defgate is None:
                self._defined_gates.append(instruction)
            elif instruction.matrix.dtype == np.complex_ or instruction.matrix.dtype == np.float_:
                if not np.allclose(existing_defgate.matrix, instruction.matrix):
                    warnings.warn('Redefining gate {}'.format(instruction.name))
                    self._defined_gates[-r_idx] = instruction
            elif not np.all(existing_defgate.matrix == instruction.matrix):
                warnings.warn('Redefining gate {}'.format(instruction.name))
                self._defined_gates[-r_idx] = instruction
        elif isinstance(instruction, DefCalibration):
            r_idx, existing_calibration = next(((i, gate) for i, gate in enumerate(reversed(self.calibrations)) if isinstance(gate, DefCalibration) and gate.name == instruction.name and (gate.parameters == instruction.parameters) and (gate.qubits == instruction.qubits)), (0, None))
            if existing_calibration is None:
                self._calibrations.append(instruction)
            elif existing_calibration.out() != instruction.out():
                warnings.warn('Redefining calibration {}'.format(instruction.name))
                self._calibrations[-r_idx] = instruction
        elif isinstance(instruction, DefMeasureCalibration):
            r_idx, existing_measure_calibration = next(((i, meas) for i, meas in enumerate(reversed(self.calibrations)) if isinstance(meas, DefMeasureCalibration) and meas.name == instruction.name and (meas.qubit == instruction.qubit)), (0, None))
            if existing_measure_calibration is None:
                self._calibrations.append(instruction)
            else:
                warnings.warn('Redefining DefMeasureCalibration {}'.format(instruction.name))
                self._calibrations[-r_idx] = instruction
        elif isinstance(instruction, DefWaveform):
            self.waveforms[instruction.name] = instruction
        elif isinstance(instruction, DefFrame):
            self.frames[instruction.frame] = instruction
        elif isinstance(instruction, AbstractInstruction):
            self._instructions.append(instruction)
            self._synthesized_instructions = None
            if isinstance(instruction, Declare):
                self._declarations[instruction.name] = instruction
        else:
            raise TypeError('Invalid instruction: {}'.format(instruction))
    return self