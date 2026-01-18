import functools
import inspect
import os
import warnings
import pennylane as qml
def tape_transform(self, fn):
    """Register a tape transformation to enable the operator transform
        to apply to datastructures containing multiple operations, such as QNodes, qfuncs,
        and tapes.

        .. note::

            The registered tape transform should have the same parameters as the
            original operation transform function.

        .. note::

            If the transformation maps a tape to a tape (or equivalently, a qfunc to a qfunc)
            then the transformation is simultaneously a :func:`~.qfunc_transform`, and
            can be declared as such. This enables additional functionality, for example
            the ability to use the transform in a compilation pipeline.

        Args:
            fn (callable): The function to register as the tape transform. This function
                should accept a :class:`~.QuantumTape` as the first argument.

        **Example**

        .. code-block:: python

            @qml.op_transform
            def name(op, lower=False):
                if lower:
                    return op.name.lower()
                return op.name

            @name.tape_transform
            def name(tape, lower=True):
                return [name(op, lower=lower) for op in tape.operations]

        We can now use this function on a qfunc, tape, or QNode:

        >>> def circuit(x, y):
        ...     qml.RX(x, wires=0)
        ...     qml.Hadamard(wires=1)
        ...     qml.CNOT(wires=[0, 1])
        ...     qml.CRY(y, wires=[1, 0])
        >>> name(circuit, lower=True)(0.1, 0.8)
        ['rx', 'hadamard', 'cnot', 'cry']

        If the transformation has purely quantum output, we can register the tape transformation
        as a qfunc transformation in addition:

        .. code-block:: python

            @qml.op_transform
            def simplify_rotation(op):
                if op.name == "Rot":
                    params = op.parameters
                    wires = op.wires

                    if qml.math.allclose(params, 0):
                        return

                    if qml.math.allclose(params[1:2], 0):
                        return qml.RZ(params[0], wires)

                return op

            @simplify_rotation.tape_transform
            @qml.qfunc_transform
            def simplify_rotation(tape):
                for op in tape:
                    if op.name == "Rot":
                        simplify_rotation(op)
                    else:
                        qml.apply(op)

        We can now use this combined operator and quantum function transform in compilation pipelines:

        .. code-block:: python

            @qml.qnode(dev)
            @qml.compile(pipeline=[simplify_rotation])
            def circuit(weights):
                ansatz(weights)
                qml.CNOT(wires=[0, 1])
                qml.Rot(0.0, 0.0, 0.0, wires=0)
                return qml.expval(qml.X(1))
        """
    self._tape_fn = fn
    return self