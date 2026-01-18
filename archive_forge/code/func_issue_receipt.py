import datetime
from oslo_log import log
from oslo_utils import timeutils
from keystone.common import cache
from keystone.common import manager
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.models import receipt_model
from keystone import notifications
def issue_receipt(self, user_id, method_names, expires_at=None):
    receipt = receipt_model.ReceiptModel()
    receipt.user_id = user_id
    receipt.methods = method_names
    if isinstance(expires_at, datetime.datetime):
        receipt.expires_at = utils.isotime(expires_at, subsecond=True)
    if isinstance(expires_at, str):
        receipt.expires_at = expires_at
    elif not expires_at:
        receipt.expires_at = utils.isotime(default_expire_time(), subsecond=True)
    receipt_id, issued_at = self.driver.generate_id_and_issued_at(receipt)
    receipt.mint(receipt_id, issued_at)
    if CONF.receipt.cache_on_issue:
        self._validate_receipt.set(receipt, RECEIPTS_REGION, receipt_id)
    return receipt