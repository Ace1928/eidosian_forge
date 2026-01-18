from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterable
import github.AdvisoryVulnerability
import github.NamedUser
from github.AdvisoryBase import AdvisoryBase
from github.AdvisoryCredit import AdvisoryCredit, Credit
from github.AdvisoryCreditDetailed import AdvisoryCreditDetailed
from github.GithubObject import Attribute, NotSet, Opt
def revoke_credit(self, login_or_user: str | github.NamedUser.NamedUser) -> None:
    """
        :calls: `PATCH /repos/{owner}/{repo}/security-advisories/:advisory_id <https://docs.github.com/en/rest/security-advisories/repository-advisories>`_
        """
    assert isinstance(login_or_user, (str, github.NamedUser.NamedUser)), login_or_user
    if isinstance(login_or_user, github.NamedUser.NamedUser):
        login_or_user = login_or_user.login
    patch_parameters = {'credits': [dict(login=credit.login, type=credit.type) for credit in self.credits if credit.login != login_or_user]}
    headers, data = self._requester.requestJsonAndCheck('PATCH', self.url, input=patch_parameters)
    self._useAttributes(data)