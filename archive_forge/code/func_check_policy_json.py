from oslo_policy import opts as policy_opts
from oslo_utils import fileutils
from oslo_upgradecheck import upgradecheck
def check_policy_json(self, conf):
    """Checks to see if policy file is JSON-formatted policy file."""
    conf.register_opts(policy_opts._options, group=policy_opts._option_group)
    msg = 'Your policy file is JSON-formatted which is deprecated. You need to switch to YAML-formatted file. Use the ``oslopolicy-convert-json-to-yaml`` tool to convert the existing JSON-formatted files to YAML in a backwards-compatible manner: https://docs.openstack.org/oslo.policy/latest/cli/oslopolicy-convert-json-to-yaml.html.'
    status = upgradecheck.Result(upgradecheck.Code.SUCCESS)
    policy_path = conf.find_file(conf.oslo_policy.policy_file)
    if policy_path and fileutils.is_json(policy_path):
        status = upgradecheck.Result(upgradecheck.Code.FAILURE, msg)
    return status