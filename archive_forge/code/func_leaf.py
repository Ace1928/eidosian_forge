from plotly.basedatatypes import BaseTraceType as _BaseTraceType
import copy as _copy
@leaf.setter
def leaf(self, val):
    self['leaf'] = val