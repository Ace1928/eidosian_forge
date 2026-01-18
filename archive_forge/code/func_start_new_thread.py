def start_new_thread(function, args, kwargs={}):
    """Dummy implementation of _thread.start_new_thread().

    Compatibility is maintained by making sure that ``args`` is a
    tuple and ``kwargs`` is a dictionary.  If an exception is raised
    and it is SystemExit (which can be done by _thread.exit()) it is
    caught and nothing is done; all other exceptions are printed out
    by using traceback.print_exc().

    If the executed function calls interrupt_main the KeyboardInterrupt will be
    raised when the function returns.

    """
    if type(args) != type(tuple()):
        raise TypeError('2nd arg must be a tuple')
    if type(kwargs) != type(dict()):
        raise TypeError('3rd arg must be a dict')
    global _main
    _main = False
    try:
        function(*args, **kwargs)
    except SystemExit:
        pass
    except:
        import traceback
        traceback.print_exc()
    _main = True
    global _interrupt
    if _interrupt:
        _interrupt = False
        raise KeyboardInterrupt