from pygments.plugin import find_plugin_styles
from pygments.util import ClassNotFound
def get_style_by_name(name):
    if name in STYLE_MAP:
        mod, cls = STYLE_MAP[name].split('::')
        builtin = 'yes'
    else:
        for found_name, style in find_plugin_styles():
            if name == found_name:
                return style
        builtin = ''
        mod = name
        cls = name.title() + 'Style'
    try:
        mod = __import__('pygments.styles.' + mod, None, None, [cls])
    except ImportError:
        raise ClassNotFound('Could not find style module %r' % mod + (builtin and ', though it should be builtin') + '.')
    try:
        return getattr(mod, cls)
    except AttributeError:
        raise ClassNotFound('Could not find style class %r in style module.' % cls)