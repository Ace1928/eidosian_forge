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
@utils.arg('-f', '--template-file', metavar='<FILE>', help=_('Path to the template.'))
@utils.arg('-e', '--environment-file', metavar='<FILE or URL>', help=_('Path to the environment, it can be specified multiple times.'), action='append')
@utils.arg('--pre-create', metavar='<RESOURCE>', default=None, action='append', help=_('Name of a resource to set a pre-create hook to. Resources in nested stacks can be set using slash as a separator: nested_stack/another/my_resource. You can use wildcards to match multiple stacks or resources: nested_stack/an*/*_resource. This can be specified multiple times'))
@utils.arg('-u', '--template-url', metavar='<URL>', help=_('URL of template.'))
@utils.arg('-o', '--template-object', metavar='<URL>', help=_('URL to retrieve template object (e.g. from swift).'))
@utils.arg('-c', '--create-timeout', metavar='<TIMEOUT>', type=int, help=_('Stack creation timeout in minutes. DEPRECATED use %(arg)s instead.') % {'arg': '--timeout'})
@utils.arg('-t', '--timeout', metavar='<TIMEOUT>', type=int, help=_('Stack creation timeout in minutes.'))
@utils.arg('-r', '--enable-rollback', default=False, action='store_true', help=_('Enable rollback on create/update failure.'))
@utils.arg('-P', '--parameters', metavar='<KEY1=VALUE1;KEY2=VALUE2...>', help=_('Parameter values used to create the stack. This can be specified multiple times, or once with parameters separated by a semicolon.'), action='append')
@utils.arg('-Pf', '--parameter-file', metavar='<KEY=FILE>', help=_('Parameter values from file used to create the stack. This can be specified multiple times. Parameter value would be the content of the file'), action='append')
@utils.arg('--poll', metavar='SECONDS', type=int, nargs='?', const=5, help=_('Poll and report events until stack completes. Optional poll interval in seconds can be provided as argument, default 5.'))
@utils.arg('name', metavar='<STACK_NAME>', help=_('Name of the stack to create.'))
@utils.arg('--tags', metavar='<TAG1,TAG2>', help=_('A list of tags to associate with the stack.'))
def do_stack_create(hc, args):
    """Create the stack."""
    show_deprecated('heat stack-create', 'openstack stack create')
    tpl_files, template = template_utils.get_template_contents(args.template_file, args.template_url, args.template_object, http.authenticated_fetcher(hc))
    env_files_list = []
    env_files, env = template_utils.process_multiple_environments_and_files(env_paths=args.environment_file, env_list_tracker=env_files_list)
    if args.create_timeout:
        logger.warning('%(arg1)s is deprecated, please use %(arg2)s instead', {'arg1': '-c/--create-timeout', 'arg2': '-t/--timeout'})
    if args.pre_create:
        template_utils.hooks_to_env(env, args.pre_create, 'pre-create')
    fields = {'stack_name': args.name, 'disable_rollback': not args.enable_rollback, 'parameters': utils.format_all_parameters(args.parameters, args.parameter_file, args.template_file, args.template_url), 'template': template, 'files': dict(list(tpl_files.items()) + list(env_files.items())), 'environment': env}
    if env_files_list:
        fields['environment_files'] = env_files_list
    if args.tags:
        fields['tags'] = args.tags
    timeout = args.timeout or args.create_timeout
    if timeout:
        fields['timeout_mins'] = timeout
    hc.stacks.create(**fields)
    do_stack_list(hc)
    if not args.poll:
        return
    show_fields = {'stack_id': args.name}
    _do_stack_show(hc, show_fields)
    stack_status, msg = event_utils.poll_for_events(hc, args.name, action='CREATE', poll_period=args.poll)
    _do_stack_show(hc, show_fields)
    if stack_status == 'CREATE_FAILED':
        raise exc.StackFailure(msg)
    print(msg)