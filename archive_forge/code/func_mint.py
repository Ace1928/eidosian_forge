from oslo_log import log
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import reflection
from keystone.common import cache
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
def mint(self, token_id, issued_at):
    """Set the ``id`` and ``issued_at`` attributes of a token.

        The process of building a token requires setting attributes about the
        authentication and authorization context, like ``user_id`` and
        ``project_id`` for example. Once a Token object accurately represents
        this information it should be "minted". Tokens are minted when they get
        an ``id`` attribute and their creation time is recorded.

        """
    self._validate_token_resources()
    self._validate_token_user()
    self._validate_system_scope()
    self._validate_domain_scope()
    self._validate_project_scope()
    self._validate_trust_scope()
    self.id = token_id
    self.issued_at = issued_at