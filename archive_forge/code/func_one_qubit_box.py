from __future__ import annotations
from sympy.core.mul import Mul
from sympy.external import import_module
from sympy.physics.quantum.gate import Gate, OneQubitGate, CGate, CGateS
def one_qubit_box(self, t, gate_idx, wire_idx):
    """Draw a box for a single qubit gate."""
    x = self._gate_grid[gate_idx]
    y = self._wire_grid[wire_idx]
    self._axes.text(x, y, t, color='k', ha='center', va='center', bbox={'ec': 'k', 'fc': 'w', 'fill': True, 'lw': self.linewidth}, size=self.fontsize)