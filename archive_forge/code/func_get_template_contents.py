from oslo_serialization import jsonutils
from urllib import parse
from urllib import request
from zunclient.common import template_format
from zunclient.common import utils
from zunclient import exceptions
from zunclient.i18n import _
def get_template_contents(template_file=None, template_url=None, files=None):
    if template_file:
        template_url = utils.normalise_file_path_to_url(template_file)
        tpl = request.urlopen(template_url).read()
    else:
        raise exceptions.CommandErrorException(_('Need to specify exactly one of %(arg1)s, %(arg2)s or %(arg3)s') % {'arg1': '--template-file', 'arg2': '--template-url'})
    if not tpl:
        raise exceptions.CommandErrorException(_('Could not fetch template from %s') % template_url)
    try:
        if isinstance(tpl, bytes):
            tpl = tpl.decode('utf-8')
        template = template_format.parse(tpl)
    except ValueError as e:
        raise exceptions.CommandErrorException(_('Error parsing template %(url)s %(error)s') % {'url': template_url, 'error': e})
    return template