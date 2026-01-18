import argparse
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
def get_required_fields(self, fields):
    fields = self.replace_columns(fields, self.replace_rules, reverse=True)
    for field, digest_fields in self.digest_fields.items():
        if field in fields:
            fields += digest_fields['depends_on']
            fields.remove(field)
    return fields