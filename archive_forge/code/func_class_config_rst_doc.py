from __future__ import annotations
import typing as t
from collections import defaultdict
from textwrap import dedent
from traitlets import HasTraits, Undefined
from traitlets.config.application import Application
from traitlets.utils.text import indent
def class_config_rst_doc(cls: type[HasTraits], trait_aliases: dict[str, t.Any]) -> str:
    """Generate rST documentation for this class' config options.

    Excludes traits defined on parent classes.
    """
    lines = []
    classname = cls.__name__
    for _, trait in sorted(cls.class_traits(config=True).items()):
        ttype = trait.__class__.__name__
        fullname = classname + '.' + (trait.name or '')
        lines += ['.. configtrait:: ' + fullname, '']
        help = trait.help.rstrip() or 'No description'
        lines.append(indent(dedent(help)) + '\n')
        if 'Enum' in ttype:
            lines.append(indent(':options: ' + ', '.join(('``%r``' % x for x in trait.values))))
        else:
            lines.append(indent(':trait type: ' + ttype))
        if interesting_default_value(trait.default_value):
            try:
                dvr = trait.default_value_repr()
            except Exception:
                dvr = None
            if dvr is not None:
                if len(dvr) > 64:
                    dvr = dvr[:61] + '...'
                dvr = dvr.replace('\\n', '\\\\n')
                lines.append(indent(':default: ``%s``' % dvr))
        if trait_aliases[fullname]:
            fmt_aliases = format_aliases(trait_aliases[fullname])
            lines.append(indent(':CLI option: ' + fmt_aliases))
        lines.append('')
    return '\n'.join(lines)