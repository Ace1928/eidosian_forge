from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable, List, Sequence, Mapping, Optional, Set, Tuple, cast
from warnings import warn
from pyquil.quil import Program
from pyquil.quilatom import ParameterDesignator, QubitDesignator, format_parameter
from pyquil.quilbase import (
def split_on_terminal_measures(program: Program) -> Tuple[List[AbstractInstruction], List[AbstractInstruction]]:
    """
    Split a program into two lists of instructions:

    1. A set of measurement instructions occuring as the final operation on their qubit.
    2. The rest.
    """
    if not any((isinstance(instr, Measurement) for instr in program.instructions)):
        return ([], program.instructions)
    seen_qubits: Set[QubitDesignator] = set()
    measures: List[AbstractInstruction] = []
    remaining: List[AbstractInstruction] = []
    in_group = False
    for instr in reversed(program.instructions):
        if not in_group and isinstance(instr, Measurement) and (instr.qubit not in seen_qubits):
            measures.insert(0, instr)
            seen_qubits.add(instr.qubit)
        else:
            remaining.insert(0, instr)
            if isinstance(instr, (Gate, ResetQubit)):
                seen_qubits |= instr.get_qubits()
            elif isinstance(instr, Pragma):
                if instr.command == PRAGMA_END_GROUP:
                    warn('Alignment of terminal MEASURE operations mayconflict with gate group declaration.')
                    in_group = True
                elif instr.command == PRAGMA_BEGIN_GROUP:
                    in_group = False
    return (measures, remaining)