from django.db.models.expressions import OrderByList
def set_source_expressions(self, exprs):
    if isinstance(exprs[-1], OrderByList):
        *exprs, self.order_by = exprs
    return super().set_source_expressions(exprs)