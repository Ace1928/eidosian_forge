from __future__ import unicode_literals
from .base import Style
from .from_dict import style_from_dict
def style_from_pygments(style_cls=pygments_DefaultStyle, style_dict=None, include_defaults=True):
    """
    Shortcut to create a :class:`.Style` instance from a Pygments style class
    and a style dictionary.

    Example::

        from prompt_toolkit.styles.from_pygments import style_from_pygments
        from pygments.styles import get_style_by_name
        style = style_from_pygments(get_style_by_name('monokai'))

    :param style_cls: Pygments style class to start from.
    :param style_dict: Dictionary for this style. `{Token: style}`.
    :param include_defaults: (`bool`) Include prompt_toolkit extensions.
    """
    assert style_dict is None or isinstance(style_dict, dict)
    assert style_cls is None or issubclass(style_cls, pygments_Style)
    styles_dict = {}
    if style_cls is not None:
        styles_dict.update(style_cls.styles)
    if style_dict is not None:
        styles_dict.update(style_dict)
    return style_from_dict(styles_dict, include_defaults=include_defaults)