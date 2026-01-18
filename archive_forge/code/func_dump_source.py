import contextlib
def dump_source(object, **kwds):
    """write object source to a NamedTemporaryFile (instead of dill.dump)
Loads with "import" or "dill.temp.load_source".  Returns the filehandle.

    >>> f = lambda x: x**2
    >>> pyfile = dill.temp.dump_source(f, alias='_f')
    >>> _f = dill.temp.load_source(pyfile)
    >>> _f(4)
    16

    >>> f = lambda x: x**2
    >>> pyfile = dill.temp.dump_source(f, dir='.')
    >>> modulename = os.path.basename(pyfile.name).split('.py')[0]
    >>> exec('from %s import f as _f' % modulename)
    >>> _f(4)
    16

Optional kwds:
    If 'alias' is specified, the object will be renamed to the given string.

    If 'prefix' is specified, the file name will begin with that prefix,
    otherwise a default prefix is used.
    
    If 'dir' is specified, the file will be created in that directory,
    otherwise a default directory is used.
    
    If 'text' is specified and true, the file is opened in text
    mode.  Else (the default) the file is opened in binary mode.  On
    some operating systems, this makes no difference.

NOTE: Keep the return value for as long as you want your file to exist !
    """
    from .source import importable, getname
    import tempfile
    kwds.setdefault('delete', True)
    kwds.pop('suffix', '')
    alias = kwds.pop('alias', '')
    name = str(alias) or getname(object)
    name = '\n#NAME: %s\n' % name
    file = tempfile.NamedTemporaryFile(suffix='.py', **kwds)
    file.write(b(''.join([importable(object, alias=alias), name])))
    file.flush()
    return file