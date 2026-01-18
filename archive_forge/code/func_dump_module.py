import re
import os
import sys
import warnings
from dill import _dill, Pickler, Unpickler
from ._dill import (
from typing import Optional, Union
import pathlib
import tempfile
def dump_module(filename: Union[str, os.PathLike]=None, module: Optional[Union[ModuleType, str]]=None, refimported: bool=False, **kwds) -> None:
    """Pickle the current state of :py:mod:`__main__` or another module to a file.

    Save the contents of :py:mod:`__main__` (e.g. from an interactive
    interpreter session), an imported module, or a module-type object (e.g.
    built with :py:class:`~types.ModuleType`), to a file. The pickled
    module can then be restored with the function :py:func:`load_module`.

    Args:
        filename: a path-like object or a writable stream. If `None`
            (the default), write to a named file in a temporary directory.
        module: a module object or the name of an importable module. If `None`
            (the default), :py:mod:`__main__` is saved.
        refimported: if `True`, all objects identified as having been imported
            into the module's namespace are saved by reference. *Note:* this is
            similar but independent from ``dill.settings[`byref`]``, as
            ``refimported`` refers to virtually all imported objects, while
            ``byref`` only affects select objects.
        **kwds: extra keyword arguments passed to :py:class:`Pickler()`.

    Raises:
       :py:exc:`PicklingError`: if pickling fails.

    Examples:

        - Save current interpreter session state:

          >>> import dill
          >>> squared = lambda x: x*x
          >>> dill.dump_module() # save state of __main__ to /tmp/session.pkl

        - Save the state of an imported/importable module:

          >>> import dill
          >>> import pox
          >>> pox.plus_one = lambda x: x+1
          >>> dill.dump_module('pox_session.pkl', module=pox)

        - Save the state of a non-importable, module-type object:

          >>> import dill
          >>> from types import ModuleType
          >>> foo = ModuleType('foo')
          >>> foo.values = [1,2,3]
          >>> import math
          >>> foo.sin = math.sin
          >>> dill.dump_module('foo_session.pkl', module=foo, refimported=True)

        - Restore the state of the saved modules:

          >>> import dill
          >>> dill.load_module()
          >>> squared(2)
          4
          >>> pox = dill.load_module('pox_session.pkl')
          >>> pox.plus_one(1)
          2
          >>> foo = dill.load_module('foo_session.pkl')
          >>> [foo.sin(x) for x in foo.values]
          [0.8414709848078965, 0.9092974268256817, 0.1411200080598672]

        - Use `refimported` to save imported objects by reference:

          >>> import dill
          >>> from html.entities import html5
          >>> type(html5), len(html5)
          (dict, 2231)
          >>> import io
          >>> buf = io.BytesIO()
          >>> dill.dump_module(buf) # saves __main__, with html5 saved by value
          >>> len(buf.getvalue()) # pickle size in bytes
          71665
          >>> buf = io.BytesIO()
          >>> dill.dump_module(buf, refimported=True) # html5 saved by reference
          >>> len(buf.getvalue())
          438

    *Changed in version 0.3.6:* Function ``dump_session()`` was renamed to
    ``dump_module()``.  Parameters ``main`` and ``byref`` were renamed to
    ``module`` and ``refimported``, respectively.

    Note:
        Currently, ``dill.settings['byref']`` and ``dill.settings['recurse']``
        don't apply to this function.
    """
    for old_par, par in [('main', 'module'), ('byref', 'refimported')]:
        if old_par in kwds:
            message = 'The argument %r has been renamed %r' % (old_par, par)
            if old_par == 'byref':
                message += " to distinguish it from dill.settings['byref']"
            warnings.warn(message + '.', PendingDeprecationWarning)
            if locals()[par]:
                raise TypeError('both %r and %r arguments were used' % (par, old_par))
    refimported = kwds.pop('byref', refimported)
    module = kwds.pop('main', module)
    from .settings import settings
    protocol = settings['protocol']
    main = module
    if main is None:
        main = _main_module
    elif isinstance(main, str):
        main = _import_module(main)
    if not isinstance(main, ModuleType):
        raise TypeError('%r is not a module' % main)
    if hasattr(filename, 'write'):
        file = filename
    else:
        if filename is None:
            filename = str(TEMPDIR / 'session.pkl')
        file = open(filename, 'wb')
    try:
        pickler = Pickler(file, protocol, **kwds)
        pickler._original_main = main
        if refimported:
            main = _stash_modules(main)
        pickler._main = main
        pickler._byref = False
        pickler._recurse = False
        pickler._session = True
        pickler._first_pass = True
        pickler._main_modified = main is not pickler._original_main
        pickler.dump(main)
    finally:
        if file is not filename:
            file.close()
    return