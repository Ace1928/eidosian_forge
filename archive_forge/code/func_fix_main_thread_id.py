def fix_main_thread_id(on_warn=lambda msg: None, on_exception=lambda msg: None, on_critical=lambda msg: None):
    import sys
    import threading
    try:
        with threading._active_limbo_lock:
            main_thread_instance = get_main_thread_instance(threading)
            if sys.platform == 'win32':
                if hasattr(threading, '_get_ident'):
                    unlikely_thread_id = threading._get_ident()
                else:
                    unlikely_thread_id = threading.get_ident()
            else:
                unlikely_thread_id = None
            main_thread_id, critical_warning = get_main_thread_id(unlikely_thread_id)
            if main_thread_id is not None:
                main_thread_id_attr = '_ident'
                if not hasattr(main_thread_instance, main_thread_id_attr):
                    main_thread_id_attr = '_Thread__ident'
                    assert hasattr(main_thread_instance, main_thread_id_attr)
                if main_thread_id != getattr(main_thread_instance, main_thread_id_attr):
                    main_thread_instance._tstate_lock = threading._allocate_lock()
                    main_thread_instance._tstate_lock.acquire()
                    threading._active.pop(getattr(main_thread_instance, main_thread_id_attr), None)
                    setattr(main_thread_instance, main_thread_id_attr, main_thread_id)
                    threading._active[getattr(main_thread_instance, main_thread_id_attr)] = main_thread_instance
        on_warn('The threading module was not imported by user code in the main thread. The debugger will attempt to work around https://bugs.python.org/issue37416.')
        if critical_warning:
            on_critical('Issue found when debugger was trying to work around https://bugs.python.org/issue37416:\n%s' % (critical_warning,))
    except:
        on_exception('Error patching main thread id.')