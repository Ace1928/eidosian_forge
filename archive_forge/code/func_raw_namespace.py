import io
import os
import sys
import unittest
import cherrypy
from cherrypy.test import helper
def raw_namespace(key, value):
    if key == 'input.map':
        handler = cherrypy.request.handler

        def wrapper():
            params = cherrypy.request.params
            for name, coercer in value.copy().items():
                try:
                    params[name] = coercer(params[name])
                except KeyError:
                    pass
            return handler()
        cherrypy.request.handler = wrapper
    elif key == 'output':
        handler = cherrypy.request.handler

        def wrapper():
            return value(handler())
        cherrypy.request.handler = wrapper