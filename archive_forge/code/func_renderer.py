import itertools
from typing import (
from zope.interface import implementer
from twisted.web.error import (
from twisted.web.iweb import IRenderable, IRequest, ITemplateLoader
@exposer
def renderer() -> None:
    """
    Decorate with L{renderer} to use methods as template render directives.

    For example::

        class Foo(Element):
            @renderer
            def twiddle(self, request, tag):
                return tag('Hello, world.')

        <div xmlns:t="http://twistedmatrix.com/ns/twisted.web.template/0.1">
            <span t:render="twiddle" />
        </div>

    Will result in this final output::

        <div>
            <span>Hello, world.</span>
        </div>
    """