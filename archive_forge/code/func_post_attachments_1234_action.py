from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def post_attachments_1234_action(self, **kw):
    attached_fake = fake_attachment
    attached_fake['status'] = 'attached'
    return (200, {}, attached_fake)