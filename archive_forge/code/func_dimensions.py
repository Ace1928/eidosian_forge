from plotly.basedatatypes import BaseTraceType as _BaseTraceType
import copy as _copy
@dimensions.setter
def dimensions(self, val):
    self['dimensions'] = val