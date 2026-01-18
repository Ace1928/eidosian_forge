import logging
from asgiref.sync import iscoroutinefunction, markcoroutinefunction
from django.core.exceptions import ImproperlyConfigured
from django.http import (
from django.template.response import TemplateResponse
from django.urls import reverse
from django.utils.decorators import classonlymethod
from django.utils.functional import classproperty
@classonlymethod
def as_view(cls, **initkwargs):
    """Main entry point for a request-response process."""
    for key in initkwargs:
        if key in cls.http_method_names:
            raise TypeError('The method name %s is not accepted as a keyword argument to %s().' % (key, cls.__name__))
        if not hasattr(cls, key):
            raise TypeError('%s() received an invalid keyword %r. as_view only accepts arguments that are already attributes of the class.' % (cls.__name__, key))

    def view(request, *args, **kwargs):
        self = cls(**initkwargs)
        self.setup(request, *args, **kwargs)
        if not hasattr(self, 'request'):
            raise AttributeError("%s instance has no 'request' attribute. Did you override setup() and forget to call super()?" % cls.__name__)
        return self.dispatch(request, *args, **kwargs)
    view.view_class = cls
    view.view_initkwargs = initkwargs
    view.__doc__ = cls.__doc__
    view.__module__ = cls.__module__
    view.__annotations__ = cls.dispatch.__annotations__
    view.__dict__.update(cls.dispatch.__dict__)
    if cls.view_is_async:
        markcoroutinefunction(view)
    return view