import sys
import importlib
import cherrypy
from cherrypy.test import helper
@classmethod
def setup_tutorial(cls, name, root_name, config={}):
    cherrypy.config.reset()
    module = cls.load_module(name)
    root = getattr(module, root_name)
    conf = getattr(module, 'tutconf')
    class_types = (type,)
    if isinstance(root, class_types):
        root = root()
    cherrypy.tree.mount(root, config=conf)
    cherrypy.config.update(config)