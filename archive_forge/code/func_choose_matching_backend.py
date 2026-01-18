import ast
from tempest.lib.cli import output_parser
import testtools
from manilaclient import api_versions
from manilaclient import config
def choose_matching_backend(share, pools, share_type):
    extra_specs = {}
    pair = [x.strip() for x in share_type['required_extra_specs'].split(':')]
    if len(pair) == 2:
        value = True if str(pair[1]).lower() == 'true' else False if str(pair[1]).lower() == 'false' else pair[1]
        extra_specs[pair[0]] = value
    selected_pool = next((x for x in pools if x['Name'] != share['host'] and all((y in ast.literal_eval(x['Capabilities']).items() for y in extra_specs.items()))), None)
    return selected_pool['Name']