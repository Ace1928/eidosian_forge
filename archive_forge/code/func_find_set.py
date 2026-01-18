from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGOpNode, DAGInNode
def find_set(self, index):
    """DSU function for finding root of set of items
        If my parent is myself, I am the root. Otherwise we recursively
        find the root for my parent. After that, we assign my parent to be
        my root, saving recursion in the future.
        """
    if index not in self.parent:
        self.parent[index] = index
        self.bit_groups[index] = [index]
        self.gate_groups[index] = []
    if self.parent[index] == index:
        return index
    self.parent[index] = self.find_set(self.parent[index])
    return self.parent[index]