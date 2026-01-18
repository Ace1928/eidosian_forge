from plotly.basedatatypes import BaseTraceType as _BaseTraceType
import copy as _copy
@geo.setter
def geo(self, val):
    self['geo'] = val