from functools import wraps
from importlib import import_module
from inspect import getfullargspec, unwrap
from django.utils.html import conditional_escape
from django.utils.itercompat import is_iterable
from .base import Node, Template, token_kwargs
from .exceptions import TemplateSyntaxError
def inclusion_tag(self, filename, func=None, takes_context=None, name=None):
    """
        Register a callable as an inclusion tag:

        @register.inclusion_tag('results.html')
        def show_results(poll):
            choices = poll.choice_set.all()
            return {'choices': choices}
        """

    def dec(func):
        params, varargs, varkw, defaults, kwonly, kwonly_defaults, _ = getfullargspec(unwrap(func))
        function_name = name or func.__name__

        @wraps(func)
        def compile_func(parser, token):
            bits = token.split_contents()[1:]
            args, kwargs = parse_bits(parser, bits, params, varargs, varkw, defaults, kwonly, kwonly_defaults, takes_context, function_name)
            return InclusionNode(func, takes_context, args, kwargs, filename)
        self.tag(function_name, compile_func)
        return func
    return dec