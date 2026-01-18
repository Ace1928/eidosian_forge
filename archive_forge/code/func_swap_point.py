from __future__ import annotations
from sympy.core.mul import Mul
from sympy.external import import_module
from sympy.physics.quantum.gate import Gate, OneQubitGate, CGate, CGateS
def swap_point(self, gate_idx, wire_idx):
    """Draw a swap point as a cross."""
    x = self._gate_grid[gate_idx]
    y = self._wire_grid[wire_idx]
    d = self.swap_delta
    l1 = Line2D((x - d, x + d), (y - d, y + d), color='k', lw=self.linewidth)
    l2 = Line2D((x - d, x + d), (y + d, y - d), color='k', lw=self.linewidth)
    self._axes.add_line(l1)
    self._axes.add_line(l2)