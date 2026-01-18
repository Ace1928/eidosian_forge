from pyglet import compat_platform
def modifiers_string(modifiers):
    """Return a string describing a set of modifiers.

    Example::

        >>> modifiers_string(MOD_SHIFT | MOD_CTRL)
        'MOD_SHIFT|MOD_CTRL'

    :Parameters:
        `modifiers` : int
            Bitwise combination of modifier constants.

    :rtype: str
    """
    mod_names = []
    if modifiers & MOD_SHIFT:
        mod_names.append('MOD_SHIFT')
    if modifiers & MOD_CTRL:
        mod_names.append('MOD_CTRL')
    if modifiers & MOD_ALT:
        mod_names.append('MOD_ALT')
    if modifiers & MOD_CAPSLOCK:
        mod_names.append('MOD_CAPSLOCK')
    if modifiers & MOD_NUMLOCK:
        mod_names.append('MOD_NUMLOCK')
    if modifiers & MOD_SCROLLLOCK:
        mod_names.append('MOD_SCROLLLOCK')
    if modifiers & MOD_COMMAND:
        mod_names.append('MOD_COMMAND')
    if modifiers & MOD_OPTION:
        mod_names.append('MOD_OPTION')
    if modifiers & MOD_FUNCTION:
        mod_names.append('MOD_FUNCTION')
    return '|'.join(mod_names)