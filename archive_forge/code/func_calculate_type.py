from keystone.assignment.backends import base
from keystone.common import sql
from keystone import exception
from keystone.i18n import _
@classmethod
def calculate_type(cls, user_id, group_id, project_id, domain_id):
    if user_id:
        if project_id:
            return cls.USER_PROJECT
        if domain_id:
            return cls.USER_DOMAIN
    if group_id:
        if project_id:
            return cls.GROUP_PROJECT
        if domain_id:
            return cls.GROUP_DOMAIN
    raise exception.AssignmentTypeCalculationError(**locals())