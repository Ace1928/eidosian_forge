import functools
from .autoray import (
from . import lazy
def autojit(fn=None, *, backend=None, compiler_opts=None):
    """Just-in-time compile an ``autoray`` function, automatically choosing
    the backend based on the input arrays, or via keyword argument.

    The backend used to do the compilation can be set in three ways:

        1. Automatically based on the arrays the function is called with,
           i.e. ``cfn(*torch_arrays)`` will use ``torch.jit.trace``.
        2. In this wrapper, ``@autojit(backend='jax')``, to provide a
           specific default instead.
        3. When you call the function ``cfn(*arrays, backend='torch')`` to
           override on a per-call basis.

    If the arrays supplied are of a different backend type to the compiler,
    then the returned array will also be converted back, i.e.
    ``cfn(*numpy_arrays, backend='tensorflow')`` will return a ``numpy`` array.

    The ``'python'`` backend simply extracts and unravels all the ``do`` calls
    into a code object using ``compile`` which is then run with ``exec``.
    This makes use of shared intermediates and constant folding, strips
    away any python scaffoliding, and is compatible with any library, but the
    resulting function is not 'low-level' in the same way as the other
    backends.

    Parameters
    ----------
    fn : callable
        The autoray function to compile.
    backend : {None, 'python', 'jax', 'torch', 'tensorflow'}, optional
        If set, use this as the default backend.
    compiler_opts : dict[dict], optional
        Dict of dicts when you can supply options for each compiler backend
        separately, e.g.:
        ``@autojit(compiler_opts={'tensorflow': {'jit_compile': True}})``.

    Returns
    -------
    cfn : callable
        The function with auto compilation.
    """
    kws = dict(backend=backend, compiler_opts=compiler_opts)
    if fn is None:
        return functools.partial(autojit, **kws)
    return functools.wraps(fn)(AutoCompiled(fn, **kws))