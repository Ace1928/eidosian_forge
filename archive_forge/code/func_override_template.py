from inspect import Arguments
from itertools import chain, tee
from mimetypes import guess_type, add_type
from os.path import splitext
import logging
import operator
import sys
import types
from webob import (Request as WebObRequest, Response as WebObResponse, exc,
from webob.multidict import NestedMultiDict
from .compat import urlparse, izip, is_bound_method as ismethod
from .jsonify import encode as dumps
from .secure import handle_security
from .templating import RendererFactory
from .routing import lookup_controller, NonCanonicalPath
from .util import _cfg, getargspec
from .middleware.recursive import ForwardRequestException
def override_template(template, content_type=None):
    """
    Call within a controller to override the template that is used in
    your response.

    :param template: a valid path to a template file, just as you would specify
                     in an ``@expose``.
    :param content_type: a valid MIME type to use for the response.func_closure
    """
    request.pecan['override_template'] = template
    if content_type:
        request.pecan['override_content_type'] = content_type