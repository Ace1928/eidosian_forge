import datetime
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resource
from oslo_log import log as logging
from oslo_utils import timeutils
def handle_metadata_reset(self):
    metadata = self.metadata_get()
    if 'scaling_in_progress' in metadata:
        metadata['scaling_in_progress'] = False
        self.metadata_set(metadata)