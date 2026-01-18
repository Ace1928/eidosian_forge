from .circuit_to_dag import circuit_to_dag
from .dag_to_circuit import dag_to_circuit
from .circuit_to_instruction import circuit_to_instruction
from .circuit_to_gate import circuit_to_gate
from .circuit_to_dagdependency import circuit_to_dagdependency
from .dagdependency_to_circuit import dagdependency_to_circuit
from .dag_to_dagdependency import dag_to_dagdependency
from .dagdependency_to_dag import dagdependency_to_dag
def isinstancelist(obj):
    """Like isinstance(obj, list), but with casting. Except for strings and dicts."""
    if isinstance(obj, (str, dict)):
        return False
    try:
        list(obj)
        return True
    except TypeError:
        return False