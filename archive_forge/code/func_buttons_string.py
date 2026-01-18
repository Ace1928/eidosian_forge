def buttons_string(buttons):
    """Return a string describing a set of active mouse buttons.

    Example::

        >>> buttons_string(LEFT | RIGHT)
        'LEFT|RIGHT'

    :Parameters:
        `buttons` : int
            Bitwise combination of mouse button constants.

    :rtype: str
    """
    button_names = []
    if buttons & LEFT:
        button_names.append('LEFT')
    if buttons & MIDDLE:
        button_names.append('MIDDLE')
    if buttons & RIGHT:
        button_names.append('RIGHT')
    if buttons & MOUSE4:
        button_names.append('MOUSE4')
    if buttons & MOUSE5:
        button_names.append('MOUSE5')
    return '|'.join(button_names)