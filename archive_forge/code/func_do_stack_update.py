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
@utils.arg('--pre-update', metavar='<RESOURCE>', default=None, action='append', help=_('Name of a resource to set a pre-update hook to. Resources in nested stacks can be set using slash as a separator: nested_stack/another/my_resource. You can use wildcards to match multiple stacks or resources: nested_stack/an*/*_resource. This can be specified multiple times'))
@utils.arg('-u', '--template-url', metavar='<URL>', help=_('URL of template.'))
@utils.arg('-o', '--template-object', metavar='<URL>', help=_('URL to retrieve template object (e.g. from swift).'))
@utils.arg('-t', '--timeout', metavar='<TIMEOUT>', type=int, help=_('Stack update timeout in minutes.'))
@utils.arg('-r', '--enable-rollback', default=False, action='store_true', help=_('DEPRECATED! Use %(arg)s argument instead. Enable rollback on stack update failure. NOTE: default behavior is now to use the rollback value of existing stack.') % {'arg': '--rollback'})
@utils.arg('--rollback', default=None, metavar='<VALUE>', help=_('Set rollback on update failure. Values %(true)s  set rollback to enabled. Values %(false)s set rollback to disabled. Default is to use the value of existing stack to be updated.') % {'true': strutils.TRUE_STRINGS, 'false': strutils.FALSE_STRINGS})
@utils.arg('-y', '--dry-run', default=False, action='store_true', help=_('Do not actually perform the stack update, but show what would be changed'))
@utils.arg('-n', '--show-nested', default=False, action='store_true', help=_('Show nested stacks when performing --dry-run'))
@utils.arg('-P', '--parameters', metavar='<KEY1=VALUE1;KEY2=VALUE2...>', help=_('Parameter values used to create the stack. This can be specified multiple times, or once with parameters separated by a semicolon.'), action='append')
@utils.arg('-Pf', '--parameter-file', metavar='<KEY=FILE>', help=_('Parameter values from file used to create the stack. This can be specified multiple times. Parameter value would be the content of the file'), action='append')
@utils.arg('-x', '--existing', default=False, action='store_true', help=_('Re-use the template, parameters and environment of the current stack. If the template argument is omitted then the existing template is used. If no %(env_arg)s is specified then the existing environment is used. Parameters specified in %(arg)s will patch over the existing values in the current stack. Parameters omitted will keep the existing values.') % {'arg': '--parameters', 'env_arg': '--environment-file'})
@utils.arg('-c', '--clear-parameter', metavar='<PARAMETER>', help=_('Remove the parameters from the set of parameters of current stack for the %(cmd)s. The default value in the template will be used. This can be specified multiple times.') % {'cmd': 'stack-update'}, action='append')
@utils.arg('id', metavar='<NAME or ID>', help=_('Name or ID of stack to update.'))
@utils.arg('--tags', metavar='<TAG1,TAG2>', help=_('An updated list of tags to associate with the stack.'))
def do_stack_update(hc, args):
    """Update the stack."""
    show_deprecated('heat stack-update', 'openstack stack update')
    tpl_files, template = template_utils.get_template_contents(args.template_file, args.template_url, args.template_object, http.authenticated_fetcher(hc), existing=args.existing)
    env_files_list = []
    env_files, env = template_utils.process_multiple_environments_and_files(env_paths=args.environment_file, env_list_tracker=env_files_list)
    if args.pre_update:
        template_utils.hooks_to_env(env, args.pre_update, 'pre-update')
    fields = {'stack_id': args.id, 'parameters': utils.format_all_parameters(args.parameters, args.parameter_file, args.template_file, args.template_url), 'existing': args.existing, 'template': template, 'files': dict(list(tpl_files.items()) + list(env_files.items())), 'environment': env}
    if env_files_list:
        fields['environment_files'] = env_files_list
    if args.tags:
        fields['tags'] = args.tags
    if args.timeout:
        fields['timeout_mins'] = args.timeout
    if args.clear_parameter:
        fields['clear_parameters'] = list(args.clear_parameter)
    if args.rollback is not None:
        try:
            rollback = strutils.bool_from_string(args.rollback, strict=True)
        except ValueError as ex:
            raise exc.CommandError(str(ex))
        else:
            fields['disable_rollback'] = not rollback
    elif args.enable_rollback:
        fields['disable_rollback'] = False
    if args.dry_run is True:
        if args.show_nested:
            fields['show_nested'] = args.show_nested
        resource_changes = hc.stacks.preview_update(**fields)
        formatters = {'resource_identity': utils.json_formatter}
        fields = ['state', 'resource_name', 'resource_type', 'resource_identity']
        for k in resource_changes.get('resource_changes', {}):
            for i in range(len(resource_changes['resource_changes'][k])):
                resource_changes['resource_changes'][k][i]['state'] = k
        utils.print_update_list(sum(resource_changes['resource_changes'].values(), []), fields, formatters=formatters)
        return
    hc.stacks.update(**fields)
    do_stack_list(hc)