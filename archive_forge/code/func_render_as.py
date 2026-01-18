from __future__ import absolute_import, unicode_literals
import itertools
import warnings
from abc import ABCMeta, abstractmethod
import six
from pybtex import textutils
from pybtex.utils import collect_iterable, deprecated
from pybtex import py3compat
def render_as(self, backend_name):
    """
        Render this :py:class:`Text` into markup.
        This is a wrapper method that loads a formatting backend plugin
        and calls :py:meth:`Text.render`.

        >>> text = Text('Longcat is ', Tag('em', 'looooooong'), '!')
        >>> print(text.render_as('html'))
        Longcat is <em>looooooong</em>!
        >>> print(text.render_as('latex'))
        Longcat is \\emph{looooooong}!
        >>> print(text.render_as('text'))
        Longcat is looooooong!

        :param backend_name: The name of the output backend (like ``"latex"`` or
            ``"html"``).

        """
    from pybtex.plugin import find_plugin
    backend_cls = find_plugin('pybtex.backends', backend_name)
    return self.render(backend_cls())