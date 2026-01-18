import inspect
import logging
from inspect import Parameter
from ray._private.inspect_util import is_cython
def flatten_args(signature_parameters: list, args, kwargs):
    """Validates the arguments against the signature and flattens them.

    The flat list representation is a serializable format for arguments.
    Since the flatbuffer representation of function arguments is a list, we
    combine both keyword arguments and positional arguments. We represent
    this with two entries per argument value - [DUMMY_TYPE, x] for positional
    arguments and [KEY, VALUE] for keyword arguments. See the below example.
    See `recover_args` for logic restoring the flat list back to args/kwargs.

    Args:
        signature_parameters: The list of Parameter objects
            representing the function signature, obtained from
            `extract_signature`.
        args: The non-keyword arguments passed into the function.
        kwargs: The keyword arguments passed into the function.

    Returns:
        List of args and kwargs. Non-keyword arguments are prefixed
            by internal enum DUMMY_TYPE.

    Raises:
        TypeError: Raised if arguments do not fit in the function signature.
    """
    reconstructed_signature = inspect.Signature(parameters=signature_parameters)
    try:
        reconstructed_signature.bind(*args, **kwargs)
    except TypeError as exc:
        raise TypeError(str(exc)) from None
    list_args = []
    for arg in args:
        list_args += [DUMMY_TYPE, arg]
    for keyword, arg in kwargs.items():
        list_args += [keyword, arg]
    return list_args