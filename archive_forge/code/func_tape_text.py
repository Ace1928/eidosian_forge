from dataclasses import dataclass
from typing import Optional
import pennylane as qml
from pennylane.measurements import (
from .drawable_layers import drawable_layers
from .utils import convert_wire_order, unwrap_controls, cwire_connections, default_bit_map
def tape_text(tape, wire_order=None, show_all_wires=False, decimals=None, max_length=100, show_matrices=True, cache=None):
    """Text based diagram for a Quantum Tape.

    Args:
        tape (QuantumTape): the operations and measurements to draw

    Keyword Args:
        wire_order (Sequence[Any]): the order (from top to bottom) to print the wires of the circuit
        show_all_wires (bool): If True, all wires, including empty wires, are printed.
        decimals (int): How many decimal points to include when formatting operation parameters.
            Default ``None`` will omit parameters from operation labels.
        max_length (Int) : Maximum length of a individual line.  After this length, the diagram will
            begin anew beneath the previous lines.
        show_matrices=True (bool): show matrix valued parameters below all circuit diagrams
        cache (dict): Used to store information between recursive calls. Necessary keys are ``'tape_offset'``
            and ``'matrices'``.

    Returns:
        str : String based graphic of the circuit.

    **Example:**

    .. code-block:: python

        ops = [
            qml.QFT(wires=(0, 1, 2)),
            qml.RX(1.234, wires=0),
            qml.RY(1.234, wires=1),
            qml.RZ(1.234, wires=2),
            qml.Toffoli(wires=(0, 1, "aux"))
        ]
        measurements = [
            qml.expval(qml.Z("aux")),
            qml.var(qml.Z(0) @ qml.Z(1)),
            qml.probs(wires=(0, 1, 2, "aux"))
        ]
        tape = qml.tape.QuantumTape(ops, measurements)

    >>> print(qml.drawer.tape_text(tape))
      0: ─╭QFT──RX─╭●─┤ ╭Var[Z@Z] ╭Probs
      1: ─├QFT──RY─├●─┤ ╰Var[Z@Z] ├Probs
      2: ─╰QFT──RZ─│──┤           ├Probs
    aux: ──────────╰X─┤  <Z>      ╰Probs

    .. details::
        :title: Usage Details

    By default, parameters are omitted. By specifying the ``decimals`` keyword, parameters
    are displayed to the specified precision. Matrix-valued parameters are never displayed.

    >>> print(qml.drawer.tape_text(tape, decimals=2))
      0: ─╭QFT──RX(1.23)─╭●─┤ ╭Var[Z@Z] ╭Probs
      1: ─├QFT──RY(1.23)─├●─┤ ╰Var[Z@Z] ├Probs
      2: ─╰QFT──RZ(1.23)─│──┤           ├Probs
    aux: ────────────────╰X─┤  <Z>      ╰Probs


    The ``max_length`` keyword wraps long circuits:

    .. code-block:: python

        rng = np.random.default_rng(seed=42)
        shape = qml.StronglyEntanglingLayers.shape(n_wires=5, n_layers=5)
        params = rng.random(shape)
        tape2 = qml.StronglyEntanglingLayers(params, wires=range(5)).expand()
        print(qml.drawer.tape_text(tape2, max_length=60))


    .. code-block:: none

        0: ──Rot─╭●──────────╭X──Rot─╭●───────╭X──Rot──────╭●────╭X
        1: ──Rot─╰X─╭●───────│───Rot─│──╭●────│──╭X────Rot─│──╭●─│─
        2: ──Rot────╰X─╭●────│───Rot─╰X─│──╭●─│──│─────Rot─│──│──╰●
        3: ──Rot───────╰X─╭●─│───Rot────╰X─│──╰●─│─────Rot─╰X─│────
        4: ──Rot──────────╰X─╰●──Rot───────╰X────╰●────Rot────╰X───

        ───Rot───────────╭●─╭X──Rot──────╭●──────────────╭X─┤
        ──╭X────Rot──────│──╰●─╭X────Rot─╰X───╭●─────────│──┤
        ──│────╭X────Rot─│─────╰●───╭X────Rot─╰X───╭●────│──┤
        ──╰●───│─────Rot─│──────────╰●───╭X────Rot─╰X─╭●─│──┤
        ───────╰●────Rot─╰X──────────────╰●────Rot────╰X─╰●─┤


    The ``wire_order`` keyword specifies the order of the wires from
    top to bottom:

    >>> print(qml.drawer.tape_text(tape, wire_order=["aux", 2, 1, 0]))
    aux: ──────────╭X─┤  <Z>      ╭Probs
      2: ─╭QFT──RZ─│──┤           ├Probs
      1: ─├QFT──RY─├●─┤ ╭Var[Z@Z] ├Probs
      0: ─╰QFT──RX─╰●─┤ ╰Var[Z@Z] ╰Probs

    If the wire order contains empty wires, they are only shown if the ``show_all_wires=True``.

    >>> print(qml.drawer.tape_text(tape, wire_order=["a", "b", "aux", 0, 1, 2], show_all_wires=True))
      a: ─────────────┤
      b: ─────────────┤
    aux: ──────────╭X─┤  <Z>      ╭Probs
      0: ─╭QFT──RX─├●─┤ ╭Var[Z@Z] ├Probs
      1: ─├QFT──RY─╰●─┤ ╰Var[Z@Z] ├Probs
      2: ─╰QFT──RZ────┤           ╰Probs

    Matrix valued parameters are always denoted by ``M`` followed by an integer corresponding to
    unique matrices.  The list of unique matrices can be printed at the end of the diagram by
    selecting ``show_matrices=True`` (the default):

    .. code-block:: python

        ops = [
            qml.QubitUnitary(np.eye(2), wires=0),
            qml.QubitUnitary(np.eye(2), wires=1)
        ]
        measurements = [qml.expval(qml.Hermitian(np.eye(4), wires=(0,1)))]
        tape = qml.tape.QuantumTape(ops, measurements)

    >>> print(qml.drawer.tape_text(tape))
    0: ──U(M0)─┤ ╭<𝓗(M1)>
    1: ──U(M0)─┤ ╰<𝓗(M1)>
    M0 =
    [[1. 0.]
    [0. 1.]]
    M1 =
    [[1. 0. 0. 0.]
    [0. 1. 0. 0.]
    [0. 0. 1. 0.]
    [0. 0. 0. 1.]]

    An existing matrix cache can be passed via the ``cache`` keyword. Note that the dictionary
    passed to ``cache`` will be modified during execution to contain any new matrices and the
    tape offset.

    >>> cache = {'matrices': [-np.eye(3)]}
    >>> print(qml.drawer.tape_text(tape, cache=cache))
    0: ──U(M1)─┤ ╭<𝓗(M2)>
    1: ──U(M1)─┤ ╰<𝓗(M2)>
    M0 =
    [[-1. -0. -0.]
    [-0. -1. -0.]
    [-0. -0. -1.]]
    M1 =
    [[1. 0.]
    [0. 1.]]
    M2 =
    [[1. 0. 0. 0.]
    [0. 1. 0. 0.]
    [0. 0. 1. 0.]
    [0. 0. 0. 1.]]
    >>> cache
    {'matrices': [tensor([[-1., -0., -0.],
        [-0., -1., -0.],
        [-0., -0., -1.]], requires_grad=True), tensor([[1., 0.],
        [0., 1.]], requires_grad=True), tensor([[1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]], requires_grad=True)], 'tape_offset': 0}

    When the provided tape has nested tapes inside, this function is called recursively.
    To maintain numbering of tapes to arbitrary levels of nesting, the ``cache`` keyword
    uses the ``"tape_offset"`` value to determine numbering. Note that the value is updated
    during the call.

    .. code-block:: python

        with qml.tape.QuantumTape() as tape:
            with qml.tape.QuantumTape() as tape_inner:
                qml.X(0)

        cache = {'tape_offset': 3}
        print(qml.drawer.tape_text(tape, cache=cache))
        print("New tape offset: ", cache['tape_offset'])


    .. code-block:: none

        0: ──Tape:3─┤

        Tape:3
        0: ──X─┤
        New tape offset:  4

    """
    cache = cache or {}
    cache.setdefault('tape_offset', 0)
    cache.setdefault('matrices', [])
    tape_cache = []
    wire_map = convert_wire_order(tape, wire_order=wire_order, show_all_wires=show_all_wires)
    bit_map = default_bit_map(tape)
    n_wires = len(wire_map)
    n_bits = len(bit_map)
    if n_wires == 0:
        return ''
    finished_lines = []
    layers = drawable_layers(tape.operations, wire_map=wire_map, bit_map=bit_map)
    final_operations_layer = len(layers) - 1
    layers += drawable_layers(tape.measurements, wire_map=wire_map, bit_map=bit_map)
    cwire_layers, _ = cwire_connections(layers, bit_map)
    wire_totals = [f'{wire}: ' for wire in wire_map]
    bit_totals = ['' for _ in range(n_bits)]
    line_length = max((len(s) for s in wire_totals))
    wire_totals = [s.rjust(line_length, ' ') for s in wire_totals]
    bit_totals = [s.rjust(line_length, ' ') for s in bit_totals]
    config = _Config(wire_map=wire_map, bit_map=bit_map, cur_layer=-1, cwire_layers=cwire_layers, decimals=decimals, cache=cache)
    for i, layer in enumerate(layers):
        w_filler = '─' if i <= final_operations_layer else ' '
        b_filler = '═' if i <= final_operations_layer else ' '
        add_fn = _add_op if i <= final_operations_layer else _add_measurement
        layer_str = [w_filler] * n_wires + [' '] * n_bits
        for b in bit_map.values():
            cur_b_filler = b_filler if min(cwire_layers[b]) < i < max(cwire_layers[b]) else ' '
            layer_str[b + n_wires] = cur_b_filler
        config.cur_layer = i
        for op in layer:
            if isinstance(op, qml.tape.QuantumScript):
                layer_str = _add_grouping_symbols(op, layer_str, config)
                label = f'Tape:{cache['tape_offset'] + len(tape_cache)}'
                for w in op.wires:
                    layer_str[wire_map[w]] += label
                tape_cache.append(op)
            else:
                layer_str = add_fn(op, layer_str, config)
        max_label_len = max((len(s) for s in layer_str))
        for w in range(n_wires):
            layer_str[w] = layer_str[w].ljust(max_label_len, w_filler)
        for b in range(n_bits):
            cur_b_filler = b_filler if cwire_layers[b][0] <= i < cwire_layers[b][-1] else ' '
            layer_str[b + n_wires] = layer_str[b + n_wires].ljust(max_label_len, cur_b_filler)
        line_length += max_label_len + 1
        if line_length > max_length:
            finished_lines += wire_totals + bit_totals
            finished_lines[-1] += '\n'
            wire_totals = [w_filler] * n_wires
            bit_totals = []
            for b in range(n_bits):
                cur_b_filler = b_filler if cwire_layers[b][0] < i <= cwire_layers[b][-1] else ' '
                bit_totals.append(cur_b_filler)
            line_length = 2 + max_label_len
        wire_totals = [w_filler.join([t, s]) for t, s in zip(wire_totals, layer_str[:n_wires])]
        for j, (bt, s) in enumerate(zip(bit_totals, layer_str[n_wires:n_wires + n_bits])):
            cur_b_filler = b_filler if cwire_layers[j][0] < i <= cwire_layers[j][-1] else ' '
            bit_totals[j] = cur_b_filler.join([bt, s])
        if i == final_operations_layer:
            wire_totals = [f'{s}─┤' for s in wire_totals]
            for b in range(n_bits):
                if cwire_layers[b][-1] > final_operations_layer:
                    bit_totals[b] += '═╡'
                else:
                    bit_totals[b] += '  '
            line_length += 2
    tape_totals = '\n'.join(finished_lines + wire_totals + bit_totals)
    current_tape_offset = cache['tape_offset']
    cache['tape_offset'] += len(tape_cache)
    for i, nested_tape in enumerate(tape_cache):
        label = f'\nTape:{i + current_tape_offset}'
        tape_str = tape_text(nested_tape, wire_order, show_all_wires, decimals, max_length, show_matrices=False, cache=cache)
        tape_totals = '\n'.join([tape_totals, label, tape_str])
    if show_matrices:
        mat_str = ''
        for i, mat in enumerate(cache['matrices']):
            mat_str += f'\nM{i} = \n{mat}'
        return tape_totals + mat_str
    return tape_totals