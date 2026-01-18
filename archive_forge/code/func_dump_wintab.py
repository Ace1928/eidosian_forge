def dump_wintab():
    """Dump WinTab info."""
    try:
        from pyglet.input.win32 import wintab
    except:
        print('WinTab not available.')
        return
    interface_name = wintab.get_interface_name()
    impl_version = wintab.get_implementation_version()
    spec_version = wintab.get_spec_version()
    print('WinTab: {0} {1}.{2} (Spec {3}.{4})'.format(interface_name, impl_version >> 8, impl_version & 255, spec_version >> 8, spec_version & 255))