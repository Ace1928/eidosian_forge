import numpy as np
import pennylane as qml
from pennylane.measurements import ProbabilityMP
def compute_jvp_single(tangent, jac):
    """Convenience function to compute the Jacobian vector product for a given
    tangent vector and a Jacobian for a single measurement tape.

    Args:
        tangent (list, tensor_like): tangent vector
        jac (tensor_like, tuple): Jacobian matrix

    Returns:
        tensor_like: the Jacobian vector product

    **Examples**

    We start with a number of examples. A more complete, technical description is given
    further below.

    1. For a single parameter and a single measurement without shape (e.g. ``expval``, ``var``):

    .. code-block:: pycon

        >>> tangent = np.array([1.0])
        >>> jac = np.array(0.2)
        >>> qml.gradients.compute_jvp_single(tangent, jac)
        np.array(0.2)

    2. For a single parameter and a single measurement with shape (e.g. ``probs``):

    .. code-block:: pycon

        >>> tangent = np.array([2.0])
        >>> jac = np.array([0.3, 0.4])
        >>> qml.gradients.compute_jvp_single(tangent, jac)
        np.array([0.6, 0.8])

    3. For multiple parameters (in this case 2 parameters) and a single measurement
       without shape (e.g. ``expval``, ``var``):

    .. code-block:: pycon

        >>> tangent = np.array([1.0, 2.0])
        >>> jac = tuple([np.array(0.1), np.array(0.2)])
        >>> qml.gradients.compute_jvp_single(tangent, jac)
        np.array(0.5)

    4. For multiple parameters (in this case 2 parameters) and a single measurement with
       shape (e.g. ``probs``):

    .. code-block:: pycon

        >>> tangent = np.array([1.0, 0.5])
        >>> jac = tuple([np.array([0.1, 0.3]), np.array([0.2, 0.4])])
        >>> qml.gradients.compute_jvp_single(tangent, jac)
        np.array([0.2, 0.5])

    .. details::
        :title: Technical description
        :href: technical-description

        There are multiple case distinctions in this function, for particular examples see above.

        - The JVP may be for one **(A)** or multiple **(B)** parameters. We call the number of
          parameters ``k``

        - The number ``R`` of tape return type dimensions may be between 0 and 3.
          We call the return type dimensions ``r_j``

        - Each parameter may have an arbitrary number ``L_i>=0`` of dimensions

        In the following, ``(a, b)`` denotes a tensor_like of shape ``(a, b)`` and ``[(a,), (b,)]``
        / ``((a,), (b,))`` denotes a ``list`` / ``tuple`` of tensors with the indicated shapes,
        respectively. Ignore the case of no trainable parameters, as it is filtered out in advance.

        For scenario **(A)**, the input shapes can be in

        .. list-table::
           :widths: 30 40 30
           :header-rows: 1

           * - ``tangent`` shape
             - ``jac`` shape
             - Comment
           * - ``(1,)`` or ``[()]`` or ``(())``
             - ``()``
             - scalar return, scalar parameter
           * - ``(1,)`` or ``[()]`` or ``(())``
             - ``(r_1,..,r_R)``
             - tensor return, scalar parameter
           * - ``[(l_1,..,l_{L_1})]`` [1]
             - ``(l_1,..,l_{L_1})``
             - scalar return, tensor parameter
           * - ``[(l_1,..,l_{L_1})]`` [1]
             - ``(r_1,..,r_R, l_1,..,l_{L_1})``
             - tensor return, tensor parameter

        [1] Note that intuitively, ``tangent`` could be allowed to be a tensor of shape
        ``(l_1,..,l_{L_1})`` without an outer list. However, this is excluded in order
        to allow for the distinction from scenario **(B)**. Internally, this input shape for
        ``tangent`` never occurs for scenario **(A)**.

        In this scenario, the tangent is reshaped into a one-dimensional tensor with shape
        ``(tangent_size,)`` and the Jacobian is reshaped to have the dimensions
        ``(r_1, ... r_R, tangent_size)``. This is followed by a ``tensordot`` contraction over the
        ``tangent_size`` axis of both tensors.

        For scenario **(B)**, the input shapes can be in

        .. list-table::
           :widths: 30 40 30
           :header-rows: 1

           * - ``tangent`` shape
             - ``jac`` shape
             - Comment
           * - ``(k,)`` or ``[(),..,()]`` or ``((),..,())``
             - ``((),..,())`` (length ``k``)
             - scalar return, ``k`` scalar parameters
           * - ``(k,)`` or ``[(),..,()]`` or ``((),..,())``
             - ``((r_1,..,r_R),..,(r_1,..,r_R))`` [1]
             - tensor return, ``k`` scalar parameters
           * - ``[(l_1,..,l_{L_1}),..,(l_1,..,l_{L_k})]``
             - ``((l_1,..,l_{L_1}),..,(l_1,..,l_{L_k}))``
             - scalar return, ``k`` tensor parameters
           * - ``[(l_1,..,l_{L_1}),..,(l_1,..,l_{L_k})]``
             - ``((r_1,..,r_R, l_1,..,l_{L_1}),..,(r_1,..,r_R, l_1,..,l_{L_k}))`` [1]
             - tensor return, ``k`` tensor parameters

        [1] Note that the return type dimensions ``(r_1,..,r_R)`` are the same for all entries
        of ``jac``, whereas the dimensions of the entries in ``tanget``, and the according
        dimensions ``(l_1,..,l_{L_k})`` of the ``jac`` entries may differ.

        In this scenario, another case separation is used: If any of the parameters is a
        tensor (i.e. not a scalar), all tangent entries are reshaped into one-dimensional
        tensors with shapes ``(tangent_size_i,)`` and then stacked into one one-dimensional tensor.
        If there are no tensor parameters, the tangent is just stacked and reshaped.
        The Jacobians are reshaped to have the dimensions ``(r_1, ... r_R, tangent_size_i)``
        and then are concatenated along their last (potentially mismatching) axis.
        This is followed by a tensordot contraction over the axes of size
        :math:`\\sum_i` ``tangent_size_i``.

    """
    if jac is None:
        return None
    single_param = not isinstance(jac, tuple)
    if single_param and jac.shape == (0,) or (not single_param and len(jac) == 0):
        return qml.math.zeros((1, 0))
    if single_param:
        tangent = qml.math.stack(tangent)
        first_tangent_ndim = len(tangent.shape[1:])
        tangent = qml.math.flatten(tangent)
        tangent_size = tangent.shape[0]
        shape = jac.shape
        new_shape = shape[:len(shape) - first_tangent_ndim] + (tangent_size,)
        jac = qml.math.cast(qml.math.convert_like(jac, tangent), tangent.dtype)
        jac = qml.math.reshape(jac, new_shape)
        return qml.math.tensordot(jac, tangent, [[-1], [0]])
    tangent_ndims = [getattr(t, 'ndim', 0) for t in tangent]
    if isinstance(tangent, (tuple, list)) and any((ndim > 0 for ndim in tangent_ndims)):
        tangent = [qml.math.flatten(t) for t in tangent]
        tangent_sizes = [t.shape[0] for t in tangent]
        tangent = qml.math.hstack(tangent)
    else:
        tangent_sizes = [1] * len(tangent)
        tangent = qml.math.stack(tangent)
    jac_shapes = [j.shape for j in jac]
    new_shapes = [shape[:len(shape) - t_ndim] + (tsize,) for shape, t_ndim, tsize in zip(jac_shapes, tangent_ndims, tangent_sizes)]
    jac = qml.math.concatenate([qml.math.reshape(j, s) for j, s in zip(jac, new_shapes)], axis=-1)
    jac = qml.math.cast(qml.math.convert_like(jac, tangent), tangent.dtype)
    return qml.math.tensordot(jac, tangent, [[-1], [0]])