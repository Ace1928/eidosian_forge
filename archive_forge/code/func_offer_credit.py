from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterable
import github.AdvisoryVulnerability
import github.NamedUser
from github.AdvisoryBase import AdvisoryBase
from github.AdvisoryCredit import AdvisoryCredit, Credit
from github.AdvisoryCreditDetailed import AdvisoryCreditDetailed
from github.GithubObject import Attribute, NotSet, Opt
def offer_credit(self, login_or_user: str | github.NamedUser.NamedUser, credit_type: str) -> None:
    """
        Offers credit to a user for a vulnerability in a repository.
        Unless you are giving credit to yourself, the user having credit offered will need to explicitly accept the credit.
        :calls: `PATCH /repos/{owner}/{repo}/security-advisories/:advisory_id <https://docs.github.com/en/rest/security-advisories/repository-advisories>`
        """
    self.offer_credits([{'login': login_or_user, 'type': credit_type}])