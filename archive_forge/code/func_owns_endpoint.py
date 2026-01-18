from __future__ import absolute_import
from functools import wraps, partial
from flask import request, url_for, current_app
from flask import abort as original_flask_abort
from flask import make_response as original_flask_make_response
from flask.views import MethodView
from flask.signals import got_request_exception
from werkzeug.datastructures import Headers
from werkzeug.exceptions import HTTPException, MethodNotAllowed, NotFound, NotAcceptable, InternalServerError
from werkzeug.wrappers import Response as ResponseBase
from flask_restful.utils import http_status_message, unpack, OrderedDict
from flask_restful.representations.json import output_json
import sys
from types import MethodType
import operator
def owns_endpoint(self, endpoint):
    """Tests if an endpoint name (not path) belongs to this Api.  Takes
        in to account the Blueprint name part of the endpoint name.

        :param endpoint: The name of the endpoint being checked
        :return: bool
        """
    if self.blueprint:
        if endpoint.startswith(self.blueprint.name):
            endpoint = endpoint.split(self.blueprint.name + '.', 1)[-1]
        else:
            return False
    return endpoint in self.endpoints