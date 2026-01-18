from oslo_reports.models import with_default_views as mwdv
from oslo_reports.views.text import generic as generic_text_views
def opt_value(opt_obj, value):
    if opt_obj['opt'].secret:
        return '***'
    else:
        return value