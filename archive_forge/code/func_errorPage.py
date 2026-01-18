from typing import cast
from twisted.web import http
from twisted.web.iweb import IRenderable, IRequest
from twisted.web.resource import IResource, Resource
from twisted.web.template import renderElement, tags
def errorPage(code: int, brief: str, detail: str) -> _ErrorPage:
    """
    Build a resource that responds to all requests with a particular HTTP
    status code and an HTML body containing some descriptive text. This is
    useful for rendering simple error pages.

    The resource dynamically handles all paths below it. Use
    L{IResource.putChild()} to override a specific path.

    @param code: An integer HTTP status code which will be used for the
        response.

    @param brief: A short string which will be included in the response
        body as the page title.

    @param detail: A longer string which will be included in the
        response body.

    @returns: An L{IResource}
    """
    return _ErrorPage(code, brief, detail)