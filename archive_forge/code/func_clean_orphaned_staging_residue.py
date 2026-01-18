import os
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import uuidutils
from glance.common import exception
from glance.common import store_utils
from glance import context
from glance.i18n import _LE
def clean_orphaned_staging_residue(self):
    try:
        files = os.listdir(staging_store_path())
    except FileNotFoundError:
        files = []
    if not files:
        return
    LOG.debug('Found %i files in staging directory for potential cleanup', len(files))
    cleaned = ignored = error = 0
    for filename in files:
        image_id = self.get_image_id(filename)
        if not image_id:
            LOG.debug('Staging directory contains unexpected non-image file %r; ignoring', filename)
            ignored += 1
            continue
        if self.is_valid_image(image_id):
            ignored += 1
            continue
        path = os.path.join(staging_store_path(), filename)
        LOG.debug('Stale staging residue found for image %(uuid)s: %(file)r; deleting now.', {'uuid': image_id, 'file': path})
        if self.delete_file(path):
            cleaned += 1
        else:
            error += 1
    LOG.debug('Cleaned %(cleaned)i stale staging files, %(ignored)i ignored (%(error)i errors)', {'cleaned': cleaned, 'ignored': ignored, 'error': error})