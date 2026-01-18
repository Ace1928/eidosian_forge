import os
from gunicorn.errors import ConfigError
from gunicorn.app.base import Application
from gunicorn import util
def load_wsgiapp(self):
    return util.import_app(self.app_uri)