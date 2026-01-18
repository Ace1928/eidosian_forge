from __future__ import annotations
from sympy.core.mul import Mul
from sympy.external import import_module
from sympy.physics.quantum.gate import Gate, OneQubitGate, CGate, CGateS
def render_label(label, inits={}):
    """Slightly more flexible way to render labels.

    >>> from sympy.physics.quantum.circuitplot import render_label
    >>> render_label('q0')
    '$\\\\left|q0\\\\right\\\\rangle$'
    >>> render_label('q0', {'q0':'0'})
    '$\\\\left|q0\\\\right\\\\rangle=\\\\left|0\\\\right\\\\rangle$'
    """
    init = inits.get(label)
    if init:
        return '$\\left|%s\\right\\rangle=\\left|%s\\right\\rangle$' % (label, init)
    return '$\\left|%s\\right\\rangle$' % label