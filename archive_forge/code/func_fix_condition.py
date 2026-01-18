from qiskit.circuit.parametertable import ParameterTable, ParameterReferences
from qiskit.exceptions import QiskitError
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
def fix_condition(op):
    original_id = id(op)
    if (out := operation_map.get(original_id)) is not None:
        return out
    condition = getattr(op, 'condition', None)
    if condition:
        reg, val = condition
        if isinstance(reg, Clbit):
            op = op.c_if(clbit_map[reg], val)
        elif reg.size == creg.size:
            op = op.c_if(creg, val)
        else:
            raise QiskitError('Cannot convert condition in circuit with multiple classical registers to instruction')
    operation_map[original_id] = op
    return op