import logging
import sys
import textwrap
import warnings
import yaml
from oslo_config import cfg
from oslo_serialization import jsonutils
import stevedore
from oslo_policy import policy
def convert_policy_json_to_yaml(args=None, conf=None):
    logging.basicConfig(level=logging.WARN)
    if conf is None:
        conf = cfg.CONF
    conf.register_cli_opts(GENERATOR_OPTS + CONVERT_OPTS)
    conf.register_opts(GENERATOR_OPTS + CONVERT_OPTS)
    conf(args)
    _check_for_namespace_opt(conf)
    _convert_policy_json_to_yaml(conf.namespace, conf.policy_file, conf.output_file)