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
@utils.arg('-f', '--definition-file', metavar='<FILE or URL>', help=_('Path to JSON/YAML containing map defining <inputs>, <outputs>, and <options>.'))
@utils.arg('-c', '--config-file', metavar='<FILE or URL>', help=_('Path to configuration script/data.'))
@utils.arg('-g', '--group', metavar='<GROUP_NAME>', default='Heat::Ungrouped', help=_('Group name of configuration tool expected by the config.'))
@utils.arg('name', metavar='<CONFIG_NAME>', help=_('Name of the configuration to create.'))
def do_config_create(hc, args):
    """Create a software configuration."""
    show_deprecated('heat config-create', 'openstack software config create')
    config = {'group': args.group, 'config': ''}
    defn = {}
    if args.definition_file:
        defn_url = utils.normalise_file_path_to_url(args.definition_file)
        defn_raw = request.urlopen(defn_url).read() or '{}'
        defn = yaml.load(defn_raw, Loader=template_format.yaml_loader)
    config['inputs'] = defn.get('inputs', [])
    config['outputs'] = defn.get('outputs', [])
    config['options'] = defn.get('options', {})
    if args.config_file:
        config_url = utils.normalise_file_path_to_url(args.config_file)
        config['config'] = request.urlopen(config_url).read()
    validate_template = {'heat_template_version': '2013-05-23', 'resources': {args.name: {'type': 'OS::Heat::SoftwareConfig', 'properties': config}}}
    hc.stacks.validate(template=validate_template)
    sc = hc.software_configs.create(name=args.name, **config)
    print(jsonutils.dumps(sc.to_dict(), indent=2))