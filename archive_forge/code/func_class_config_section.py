from __future__ import annotations
import logging
import typing as t
from copy import deepcopy
from textwrap import dedent
from traitlets.traitlets import (
from traitlets.utils import warnings
from traitlets.utils.bunch import Bunch
from traitlets.utils.text import indent, wrap_paragraphs
from .loader import Config, DeferredConfig, LazyConfigValue, _is_section_key
@classmethod
def class_config_section(cls, classes: t.Sequence[type[HasTraits]] | None=None) -> str:
    """Get the config section for this class.

        Parameters
        ----------
        classes : list, optional
            The list of other classes in the config file.
            Used to reduce redundant information.
        """

    def c(s: str) -> str:
        """return a commented, wrapped block."""
        s = '\n\n'.join(wrap_paragraphs(s, 78))
        return '## ' + s.replace('\n', '\n#  ')
    breaker = '#' + '-' * 78
    parent_classes = ', '.join((p.__name__ for p in cls.__bases__ if issubclass(p, Configurable)))
    s = f'# {cls.__name__}({parent_classes}) configuration'
    lines = [breaker, s, breaker]
    desc = cls.class_traits().get('description')
    if desc:
        desc = desc.default_value
    if not desc:
        desc = getattr(cls, '__doc__', '')
    if desc:
        lines.append(c(desc))
        lines.append('')
    for name, trait in sorted(cls.class_traits(config=True).items()):
        default_repr = trait.default_value_repr()
        if classes:
            defining_class = cls._defining_class(trait, classes)
        else:
            defining_class = cls
        if defining_class is cls:
            if trait.help:
                lines.append(c(trait.help))
            if 'Enum' in type(trait).__name__:
                lines.append('#  Choices: %s' % trait.info())
            lines.append('#  Default: %s' % default_repr)
        else:
            if trait.help:
                lines.append(c(trait.help.split('\n', 1)[0]))
            lines.append(f'#  See also: {defining_class.__name__}.{name}')
        lines.append(f'# c.{cls.__name__}.{name} = {default_repr}')
        lines.append('')
    return '\n'.join(lines)