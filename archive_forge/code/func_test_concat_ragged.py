import itertools
import os
import time
from collections import defaultdict
from random import randint, random, sample, randrange
from typing import Iterator, Optional, Tuple, TYPE_CHECKING
import numpy as np
import pytest
import sympy
import cirq
from cirq import circuits
from cirq import ops
from cirq.testing.devices import ValidatingTestDevice
def test_concat_ragged():
    a, b = cirq.LineQubit.range(2)
    empty = cirq.Circuit()
    assert cirq.Circuit.concat_ragged(empty, empty) == empty
    assert cirq.Circuit.concat_ragged() == empty
    assert empty.concat_ragged(empty) == empty
    assert empty.concat_ragged(empty, empty) == empty
    ha = cirq.Circuit(cirq.H(a))
    hb = cirq.Circuit(cirq.H(b))
    assert ha.concat_ragged(hb) == ha.zip(hb)
    assert ha.concat_ragged(empty) == ha
    assert empty.concat_ragged(ha) == ha
    hac = cirq.Circuit(cirq.H(a), cirq.CNOT(a, b))
    assert hac.concat_ragged(hb) == hac + hb
    assert hb.concat_ragged(hac) == hb.zip(hac)
    zig = cirq.Circuit(cirq.H(a), cirq.CNOT(a, b), cirq.H(b))
    assert zig.concat_ragged(zig) == cirq.Circuit(cirq.H(a), cirq.CNOT(a, b), cirq.Moment(cirq.H(a), cirq.H(b)), cirq.CNOT(a, b), cirq.H(b))
    zag = cirq.Circuit(cirq.H(a), cirq.H(a), cirq.CNOT(a, b), cirq.H(b), cirq.H(b))
    assert zag.concat_ragged(zag) == cirq.Circuit(cirq.H(a), cirq.H(a), cirq.CNOT(a, b), cirq.Moment(cirq.H(a), cirq.H(b)), cirq.Moment(cirq.H(a), cirq.H(b)), cirq.CNOT(a, b), cirq.H(b), cirq.H(b))
    space = cirq.Circuit(cirq.Moment()) * 10
    f = cirq.Circuit.concat_ragged
    assert len(f(space, ha)) == 10
    assert len(f(space, ha, ha, ha)) == 10
    assert len(f(space, f(ha, ha, ha))) == 10
    assert len(f(space, ha, align='LEFT')) == 10
    assert len(f(space, ha, ha, ha, align='RIGHT')) == 12
    assert len(f(space, f(ha, ha, ha, align='LEFT'))) == 10
    assert len(f(space, f(ha, ha, ha, align='RIGHT'))) == 10
    assert len(f(space, f(ha, ha, ha), align='LEFT')) == 10
    assert len(f(space, f(ha, ha, ha), align='RIGHT')) == 10
    assert 7 == len(f(cirq.Circuit(cirq.CZ(a, b), [cirq.H(a)] * 5), cirq.Circuit([cirq.H(b)] * 5, cirq.CZ(a, b))))
    assert 7 == len(f(cirq.Circuit(cirq.CZ(a, b), [cirq.H(a)] * 4), cirq.Circuit([cirq.H(b)] * 5, cirq.CZ(a, b))))
    assert 7 == len(f(cirq.Circuit(cirq.CZ(a, b), [cirq.H(a)] * 1), cirq.Circuit([cirq.H(b)] * 5, cirq.CZ(a, b))))
    assert 8 == len(f(cirq.Circuit(cirq.CZ(a, b), [cirq.H(a)] * 6), cirq.Circuit([cirq.H(b)] * 5, cirq.CZ(a, b))))
    assert 9 == len(f(cirq.Circuit(cirq.CZ(a, b), [cirq.H(a)] * 7), cirq.Circuit([cirq.H(b)] * 5, cirq.CZ(a, b))))
    assert 7 == len(f(cirq.Circuit(cirq.CZ(a, b), [cirq.H(a)] * 5), cirq.Circuit([cirq.H(b)] * 5, cirq.CZ(a, b))))
    assert 7 == len(f(cirq.Circuit(cirq.CZ(a, b), [cirq.H(a)] * 5), cirq.Circuit([cirq.H(b)] * 4, cirq.CZ(a, b))))
    assert 7 == len(f(cirq.Circuit(cirq.CZ(a, b), [cirq.H(a)] * 5), cirq.Circuit([cirq.H(b)] * 1, cirq.CZ(a, b))))
    assert 8 == len(f(cirq.Circuit(cirq.CZ(a, b), [cirq.H(a)] * 5), cirq.Circuit([cirq.H(b)] * 6, cirq.CZ(a, b))))
    assert 9 == len(f(cirq.Circuit(cirq.CZ(a, b), [cirq.H(a)] * 5), cirq.Circuit([cirq.H(b)] * 7, cirq.CZ(a, b))))
    assert 10 == len(f(cirq.Circuit(cirq.Moment(), cirq.Moment(), cirq.Moment(), cirq.Moment(), cirq.Moment(), cirq.Moment(cirq.H(a)), cirq.Moment(), cirq.Moment(), cirq.Moment(cirq.H(b))), cirq.Circuit(cirq.Moment(), cirq.Moment(), cirq.Moment(), cirq.Moment(cirq.H(a)), cirq.Moment(), cirq.Moment(cirq.H(b)))))
    for cz_order in [cirq.CZ(a, b), cirq.CZ(b, a)]:
        assert 3 == len(f(cirq.Circuit(cirq.Moment(cz_order), cirq.Moment(), cirq.Moment()), cirq.Circuit(cirq.Moment(cirq.H(a)), cirq.Moment(cirq.H(b)))))
    v = ha.freeze().concat_ragged(empty)
    assert type(v) is cirq.FrozenCircuit and v == ha.freeze()
    v = ha.concat_ragged(empty.freeze())
    assert type(v) is cirq.Circuit and v == ha
    v = ha.freeze().concat_ragged(empty)
    assert type(v) is cirq.FrozenCircuit and v == ha.freeze()
    v = cirq.Circuit.concat_ragged(ha, empty)
    assert type(v) is cirq.Circuit and v == ha
    v = cirq.FrozenCircuit.concat_ragged(ha, empty)
    assert type(v) is cirq.FrozenCircuit and v == ha.freeze()