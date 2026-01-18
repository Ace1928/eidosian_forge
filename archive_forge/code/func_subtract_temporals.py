import uuid
from django.conf import settings
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.backends.utils import split_tzname_delta
from django.db.models import Exists, ExpressionWrapper, Lookup
from django.db.models.constants import OnConflict
from django.utils import timezone
from django.utils.encoding import force_str
from django.utils.regex_helper import _lazy_re_compile
def subtract_temporals(self, internal_type, lhs, rhs):
    lhs_sql, lhs_params = lhs
    rhs_sql, rhs_params = rhs
    if internal_type == 'TimeField':
        if self.connection.mysql_is_mariadb:
            return ('CAST((TIME_TO_SEC(%(lhs)s) - TIME_TO_SEC(%(rhs)s)) * 1000000 AS SIGNED)' % {'lhs': lhs_sql, 'rhs': rhs_sql}, (*lhs_params, *rhs_params))
        return ('((TIME_TO_SEC(%(lhs)s) * 1000000 + MICROSECOND(%(lhs)s)) - (TIME_TO_SEC(%(rhs)s) * 1000000 + MICROSECOND(%(rhs)s)))' % {'lhs': lhs_sql, 'rhs': rhs_sql}, tuple(lhs_params) * 2 + tuple(rhs_params) * 2)
    params = (*rhs_params, *lhs_params)
    return ('TIMESTAMPDIFF(MICROSECOND, %s, %s)' % (rhs_sql, lhs_sql), params)