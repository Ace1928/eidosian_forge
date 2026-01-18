import logging
import sys
from oslo_serialization import jsonutils
from oslo_utils import strutils
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import deployment_utils
from heatclient.common import event_utils
from heatclient.common import hook_utils
from heatclient.common import http
from heatclient.common import template_format
from heatclient.common import template_utils
from heatclient.common import utils
import heatclient.exc as exc
@utils.arg('-u', '--template-url', metavar='<URL>', help=_('URL of template.'))
@utils.arg('-f', '--template-file', metavar='<FILE>', help=_('Path to the template.'))
@utils.arg('-e', '--environment-file', metavar='<FILE or URL>', help=_('Path to the environment, it can be specified multiple times.'), action='append')
@utils.arg('-o', '--template-object', metavar='<URL>', help=_('URL to retrieve template object (e.g. from swift).'))
@utils.arg('-n', '--show-nested', default=False, action='store_true', help=_('Resolve parameters from nested templates as well.'))
@utils.arg('-P', '--parameters', metavar='<KEY1=VALUE1;KEY2=VALUE2...>', help=_('Parameter values for the template. This can be specified multiple times, or once with parameters separated by a semicolon.'), action='append')
@utils.arg('-I', '--ignore-errors', metavar='<ERR1,ERR2...>', help=_('List of heat errors to ignore.'))
def do_template_validate(hc, args):
    """Validate a template with parameters."""
    show_deprecated('heat template-validate', 'openstack orchestration template validate')
    tpl_files, template = template_utils.get_template_contents(args.template_file, args.template_url, args.template_object, http.authenticated_fetcher(hc))
    env_files_list = []
    env_files, env = template_utils.process_multiple_environments_and_files(env_paths=args.environment_file, env_list_tracker=env_files_list)
    fields = {'template': template, 'parameters': utils.format_parameters(args.parameters), 'files': dict(list(tpl_files.items()) + list(env_files.items())), 'environment': env}
    if args.ignore_errors:
        fields['ignore_errors'] = args.ignore_errors
    if env_files_list:
        fields['environment_files'] = env_files_list
    if args.show_nested:
        fields['show_nested'] = args.show_nested
    validation = hc.stacks.validate(**fields)
    print(jsonutils.dumps(validation, indent=2, ensure_ascii=False))