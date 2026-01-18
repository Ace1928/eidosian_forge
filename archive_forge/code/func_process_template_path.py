from collections import abc
from oslo_serialization import jsonutils
from urllib import error
from urllib import parse
from urllib import request
from heatclient._i18n import _
from heatclient.common import environment_format
from heatclient.common import template_format
from heatclient.common import utils
from heatclient import exc
def process_template_path(template_path, object_request=None, existing=False, fetch_child=True):
    """Read template from template path.

    Attempt to read template first as a file or url. If that is unsuccessful,
    try again assuming path is to a template object.

    :param template_path: local or uri path to template
    :param object_request: custom object request function used to get template
                           if local or uri path fails
    :param existing: if the current stack's template should be used
    :param fetch_child: Whether to fetch the child templates
    :returns: get_file dict and template contents
    :raises: error.URLError
    """
    try:
        return get_template_contents(template_file=template_path, existing=existing, fetch_child=fetch_child)
    except error.URLError as template_file_exc:
        try:
            return get_template_contents(template_object=template_path, object_request=object_request, existing=existing, fetch_child=fetch_child)
        except exc.HTTPNotFound:
            raise template_file_exc