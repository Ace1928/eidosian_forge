import copy
from oslo_reports.models import base as base_model
from oslo_reports.views.json import generic as jsonviews
from oslo_reports.views.text import generic as textviews
from oslo_reports.views.xml import generic as xmlviews
def set_current_view_type(self, tp, visited=None):
    self.attached_view = self.views[tp]
    super(ModelWithDefaultViews, self).set_current_view_type(tp, visited)