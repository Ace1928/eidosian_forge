from typing import Any, cast, Dict, List, Optional, TYPE_CHECKING, Union
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
@exc.on_http_error(exc.GitlabUpdateError)
def set_approvers(self, approvals_required: int, approver_ids: Optional[List[int]]=None, approver_group_ids: Optional[List[int]]=None, approval_rule_name: str='name', **kwargs: Any) -> RESTObject:
    """Change MR-level allowed approvers and approver groups.

        Args:
            approvals_required: The number of required approvals for this rule
            approver_ids: User IDs that can approve MRs
            approver_group_ids: Group IDs whose members can approve MRs

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabUpdateError: If the server failed to perform the request
        """
    approver_ids = approver_ids or []
    approver_group_ids = approver_group_ids or []
    data = {'name': approval_rule_name, 'approvals_required': approvals_required, 'rule_type': 'regular', 'user_ids': approver_ids, 'group_ids': approver_group_ids}
    if TYPE_CHECKING:
        assert self._parent is not None
    approval_rules: ProjectMergeRequestApprovalRuleManager = self._parent.approval_rules
    existing_approval_rules = approval_rules.list()
    for ar in existing_approval_rules:
        if ar.name == approval_rule_name:
            ar.user_ids = data['user_ids']
            ar.approvals_required = data['approvals_required']
            ar.group_ids = data['group_ids']
            ar.save()
            return ar
    return approval_rules.create(data=data, **kwargs)