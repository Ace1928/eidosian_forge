from copy import deepcopy
from flask import current_app, request
from werkzeug.datastructures import MultiDict, FileStorage
from werkzeug import exceptions
import flask_restful
import decimal
import six
def remove_argument(self, name):
    """ Remove the argument matching the given name. """
    for index, arg in enumerate(self.args[:]):
        if name == arg.name:
            del self.args[index]
            break
    return self