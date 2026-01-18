import functools
import inspect
import os
import warnings
import pennylane as qml
@property
def is_qfunc_transform(self):
    """bool: Returns ``True`` if the operator transform is also a qfunc transform.
        That is, it maps one or more quantum operations to one or more quantum operations, allowing
        the output of the transform to be used as a quantum function.

        .. seealso:: :func:`~.qfunc_transform`
        """
    return isinstance(getattr(self._tape_fn, 'tape_fn', None), qml.single_tape_transform)