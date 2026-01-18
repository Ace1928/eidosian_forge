import flask
from flask import request
import http.client
from oslo_serialization import jsonutils
from keystone.common import json_home
import keystone.conf
from keystone.server import flask as ks_flask
@_DISCOVERY_BLUEPRINT.route('/v3')
def get_version_v3():
    if v3_mime_type_best_match() == MimeTypes.JSON_HOME:
        content = json_home.JsonHomeResources.resources()
        return flask.Response(response=jsonutils.dumps(content), mimetype=MimeTypes.JSON_HOME)
    else:
        identity_url = '%s/' % ks_flask.base_url()
        versions = _get_versions_list(identity_url)
        return flask.Response(response=jsonutils.dumps({'version': versions['v3']}), mimetype=MimeTypes.JSON)