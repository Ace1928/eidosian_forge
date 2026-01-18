import inspect
import logging
import operator
import types
def get_callable_args(function, required_only=False):
    """Get names of callable arguments.

    Special arguments (like ``*args`` and ``**kwargs``) are not included into
    output.

    If required_only is True, optional arguments (with default values)
    are not included into output.
    """
    sig = get_signature(function)
    function_args = list(sig.parameters.keys())
    for param_name, p in sig.parameters.items():
        if p.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD) or (required_only and p.default is not Parameter.empty):
            function_args.remove(param_name)
    return function_args