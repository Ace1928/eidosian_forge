from plotly.basedatatypes import BaseTraceType as _BaseTraceType
import copy as _copy
@error_y.setter
def error_y(self, val):
    self['error_y'] = val