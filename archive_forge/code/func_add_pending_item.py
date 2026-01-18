from .. import ui
from ..branch import Branch
from ..check import Check
from ..i18n import gettext
from ..revision import NULL_REVISION
from ..trace import note
from ..workingtree import WorkingTree
def add_pending_item(self, referer, key, kind, sha1):
    """Add a reference to a sha1 to be cross checked against a key.

        :param referer: The referer that expects key to have sha1.
        :param key: A storage key e.g. ('texts', 'foo@bar-20040504-1234')
        :param kind: revision/inventory/text/map/signature
        :param sha1: A hex sha1 or None if no sha1 is known.
        """
    existing = self.pending_keys.get(key)
    if existing:
        if sha1 != existing[1]:
            self._report_items.append(gettext('Multiple expected sha1s for {0}. {{{1}}} expects {{{2}}}, {{{3}}} expects {{{4}}}').format(key, referer, sha1, existing[1], existing[0]))
    else:
        self.pending_keys[key] = (kind, sha1, referer)