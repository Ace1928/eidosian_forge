import logging
from os_ken.services.protocols.bgp.operator.views import fields
def combine_related(self, field_name):
    return CombinedViewsWrapper(list(_flatten([obj.combine_related(field_name) for obj in self._obj])))