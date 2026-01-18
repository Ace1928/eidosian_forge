import argparse
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
def setup_columns(self, info, parsed_args):
    parsed_args.columns = self.replace_columns(parsed_args.columns, self.replace_rules, reverse=True)
    info = super(ListSecurityGroupRule, self).setup_columns(info, parsed_args)
    cols = info[0]
    if not parsed_args.no_nameconv:
        cols = self.replace_columns(info[0], self.replace_rules)
        parsed_args.columns = cols
    return (cols, info[1])