def dump_window():
    """Dump display, window, screen and default config info."""
    from pyglet.gl import gl_info
    if not gl_info.have_version(3):
        print(f'Insufficient OpenGL version: {gl_info.get_version_string()}')
        return
    import pyglet.window
    display = pyglet.canvas.get_display()
    print('display:', repr(display))
    screens = display.get_screens()
    for i, screen in enumerate(screens):
        print(f'screens[{i}]: {screen!r}')
    window = pyglet.window.Window(visible=False)
    for key, value in window.config.get_gl_attributes():
        print(f"config['{key}'] = {value!r}")
    print('context:', repr(window.context))
    _heading('window.context._info')
    dump_gl(window.context)
    window.close()