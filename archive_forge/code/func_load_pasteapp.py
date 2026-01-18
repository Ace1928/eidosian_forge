import os
from gunicorn.errors import ConfigError
from gunicorn.app.base import Application
from gunicorn import util
def load_pasteapp(self):
    from .pasterapp import get_wsgi_app
    return get_wsgi_app(self.app_uri, defaults=self.cfg.paste_global_conf)