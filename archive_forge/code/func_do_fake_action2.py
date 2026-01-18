from cinderclient import api_versions
from cinderclient import utils
@utils.arg('--foo', start_version='3.1', end_version='3.2')
@utils.arg('--bar', help='bar help', start_version='3.3', end_version='3.4')
def do_fake_action2():
    return 'fake_action2'