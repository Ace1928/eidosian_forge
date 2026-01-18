import configparser
import os
from paste.deploy import loadapp
from gunicorn.app.wsgiapp import WSGIApplication
from gunicorn.config import get_default_config_file
def get_wsgi_app(config_uri, name=None, defaults=None):
    if ':' not in config_uri:
        config_uri = 'config:%s' % config_uri
    return loadapp(config_uri, name=name, relative_to=os.getcwd(), global_conf=defaults)